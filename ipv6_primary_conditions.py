# ipv6_primary_conditions.py
# ============================================================
# CODE 4 — IPv6 Primary Conditions (SEPARATE from baseline model features)
# ------------------------------------------------------------
# ✅ DOES NOT modify or depend on your baseline (model-trained) feature logic.
# ✅ Computes the FULL IPv6/tunnel/unmask columns used in your "effective combined primary conditions".
# ✅ Designed to be plugged into app.py later, alongside:
#    - primary keywords PKL rules
#    - HTTP primary conditions
#    - whitelist domains logic
#
# INPUT (minimum):
#   df with at least one raw column + client_ip/dest_ip (if missing, safe defaults)
#   Supported raw cols: raw_log, raw, line, message (first found)
#
# OUTPUT:
#   Returns df copy with these columns (always present; no missingness):
#     ipv6_present_raw, raw_ipv6_count, raw_ipv6_first,
#     dns_answer_ip, dns_answer_is_ipv6, aaaa_query,
#     client_ipv6_unmasked_flag, client_ipv6_unmasked, client_ipv6_type_id,
#     dest_ipv6_unmasked_flag, dest_ipv6_unmasked, dest_ipv6_type_id,
#     tunnel_6to4_flag, tunnel_teredo_flag, tunnel_isatap_flag, tunnel_nat64_flag,
#     tunnel_v4mapped_flag, tunnel_ula_flag, tunnel_linklocal_flag,
#     tunnel_proto41_hint, tunnel_udp3544_hint, tunnel_keywords_hint,
#     ip_ipv6_event_rate, ip_ipv6_burst_hour, ip_aaaa_burst_hour, ip_ipv6_switch_rate,
#     ipv6_tunnel_score, ipv6_tunnel_level_int, tunnel_exfil_interaction,
#     ipv6_primary_flag
#
# OPTIONAL INPUT COLUMNS (if present, used to improve scoring, but not required):
#   timestamp, high_bytes_out, threat_level_int, command_risk_score, session_multi_device_risk, ua_release_anomaly_flag
# ============================================================

from __future__ import annotations

import re
import ipaddress
from functools import lru_cache
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Regex / constants
# ----------------------------
_IP_TOKEN_RE = re.compile(r"(?i)(?:\[[0-9a-f:.%]{2,80}\]|[0-9a-f:.%]{2,80})(?::\d{1,5})?")
_IPV6_HINT_RE = re.compile(r"(?i)(?:::)|(?:[0-9a-f]{1,4}:){3,}[0-9a-f]{0,4}")

_DNS_ANSWER_RE = re.compile(r"\banswer=([^\s]+)")
_DNS_QTYPE_AAAA_RE = re.compile(r"\bqtype=AAAA\b|\btype=AAAA\b", re.IGNORECASE)

_TUNNEL_KEYWORDS_RE = re.compile(
    r"(?:\bteredo\b|\b6to4\b|\bisatap\b|\b6in4\b|\btunnel\b|\btunnelbroker\b|\bhe\.net\b|\bhurricane\s*electric\b)",
    re.IGNORECASE,
)
_PROTO41_RE = re.compile(
    r"(?:\bproto(?:col)?\s*=\s*41\b|\bipproto\s*=\s*41\b|\bprotocol\s+41\b|\b6in4\b|\bsit0?\b)",
    re.IGNORECASE,
)
_UDP3544_RE = re.compile(r"(?:\b3544\b|\bteredo\b)", re.IGNORECASE)

NET_6TO4 = ipaddress.ip_network("2002::/16")
NET_TEREDO = ipaddress.ip_network("2001:0000::/32")
NAT64_WKP = ipaddress.ip_network("64:ff9b::/96")
NAT64_LOCAL = ipaddress.ip_network("64:ff9b:1::/48")
NET_ULA = ipaddress.ip_network("fc00::/7")
NET_LL = ipaddress.ip_network("fe80::/10")

_IPV6_TYPE_TO_ID = {
    "none": 0,
    "native": 1,
    "v4mapped": 2,
    "v4compat": 3,
    "6to4": 4,
    "teredo": 5,
    "isatap": 6,
    "nat64": 7,
}


# ----------------------------
# Small helpers
# ----------------------------
@lru_cache(maxsize=200000)
def _try_ip(s: str):
    try:
        return ipaddress.ip_address(s)
    except Exception:
        return None


def _candidate_ip_forms(tok: str) -> list[str]:
    t = (tok or "").strip()
    if not t:
        return []

    # [IPv6]:port
    if t.startswith("["):
        rb = t.find("]")
        if rb != -1:
            return [t[1:rb].strip()]

    # IPv4:port
    if t.count(":") == 1 and "." in t:
        host, port = t.rsplit(":", 1)
        if port.isdigit():
            return [host.strip(), t]

    # IPv6:port without brackets (best-effort)
    out = [t]
    m = re.match(r"^(.*):(\d{1,5})$", t)
    if m and t.count(":") >= 2:
        out.append(m.group(1).strip())
    return out


def _extract_valid_ips_from_raw(raw_line: str) -> tuple[list[str], list[str]]:
    v4s: list[str] = []
    v6s: list[str] = []
    if not raw_line:
        return v4s, v6s

    seen4, seen6 = set(), set()
    for m in _IP_TOKEN_RE.finditer(raw_line):
        tok = m.group(0)
        for cand in _candidate_ip_forms(tok):
            ip = _try_ip(cand)
            if ip is None:
                continue
            if ip.version == 4:
                s = str(ip)
                if s not in seen4:
                    seen4.add(s)
                    v4s.append(s)
            else:
                s = str(ip)
                if s not in seen6:
                    seen6.add(s)
                    v6s.append(s)
            break
    return v4s, v6s


def _ipv6_embedded_v4_and_type(ip6: ipaddress.IPv6Address) -> tuple[list[str], str]:
    # IPv4-mapped ::ffff:a.b.c.d
    try:
        if ip6.ipv4_mapped is not None:
            return [str(ip6.ipv4_mapped)], "v4mapped"
    except Exception:
        pass

    # IPv4-compatible ::a.b.c.d (deprecated)
    try:
        if ip6 in ipaddress.ip_network("::/96") and int(ip6) <= 0xFFFFFFFF:
            return [str(ipaddress.IPv4Address(int(ip6) & 0xFFFFFFFF))], "v4compat"
    except Exception:
        pass

    # 6to4
    try:
        if ip6.sixtofour is not None:
            return [str(ip6.sixtofour)], "6to4"
    except Exception:
        pass

    # Teredo
    try:
        ter = ip6.teredo
        if ter is not None:
            server, _flags, _port, client = ter
            return [str(server), str(client)], "teredo"
    except Exception:
        pass

    # NAT64 WKP/local
    try:
        if (ip6 in NAT64_WKP) or (ip6 in NAT64_LOCAL):
            return [str(ipaddress.IPv4Address(int(ip6) & 0xFFFFFFFF))], "nat64"
    except Exception:
        pass

    # ISATAP heuristic
    try:
        if "5efe" in ip6.exploded.lower():
            return [str(ipaddress.IPv4Address(int(ip6) & 0xFFFFFFFF))], "isatap"
    except Exception:
        pass

    return [], "native"


def _unmask_ipv6_from_ipv4(ip4: str, ipv6_list: list[str]) -> tuple[str, int]:
    if not ip4 or not ipv6_list:
        return "", 0
    for v6s in ipv6_list:
        ip = _try_ip(v6s)
        if ip is None or getattr(ip, "version", 0) != 6:
            continue
        emb, typ = _ipv6_embedded_v4_and_type(ip)  # type: ignore[arg-type]
        if ip4 in emb:
            return v6s, int(_IPV6_TYPE_TO_ID.get(typ, 0))
    return "", 0


def _normalize_ip_token_series(tokens: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = tokens.fillna("").astype(str).str.strip().str.strip('",')
    codes, uniq = pd.factorize(s, sort=False)
    norm = np.empty(len(uniq), dtype=object)
    ver = np.zeros(len(uniq), dtype=np.int8)
    for i, u in enumerate(uniq):
        if not u:
            norm[i] = ""
            ver[i] = 0
            continue
        ip = _try_ip(u)
        if ip is None:
            norm[i] = ""
            ver[i] = 0
        else:
            norm[i] = str(ip)
            ver[i] = 6 if ip.version == 6 else 4
    return pd.Series(norm[codes], index=s.index, dtype="object"), pd.Series(ver[codes], index=s.index, dtype="int8")


def _pick_raw_col(df: pd.DataFrame) -> str:
    for c in ("raw_log", "raw", "line", "message", "text", "log"):
        if c in df.columns:
            return c
    # create one
    return ""


def _safe_float_col(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=np.float32)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=np.float32, copy=False)


def _safe_int_col(df: pd.DataFrame, col: str, default: int = 0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=np.int32)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=np.int32, copy=False)


# ----------------------------
# Main: compute IPv6 columns + score/flag
# ----------------------------
def add_ipv6_primary_conditions(
    df_in: pd.DataFrame,
    *,
    raw_col: Optional[str] = None,
    score_level2_threshold: float = 0.45,
    score_level3_threshold: float = 0.75,
    score_level1_threshold: float = 0.20,
    primary_level_threshold: int = 2,
) -> pd.DataFrame:
    """
    Adds IPv6/tunnel/unmask columns + ipv6_tunnel_score + ipv6_primary_flag.

    NOTE:
      - This module intentionally does NOT alter baseline model features.
      - You can combine ipv6_primary_flag into your overall primary_conditions later.

    primary flag default:
      ipv6_primary_flag = (ipv6_tunnel_level_int >= primary_level_threshold)
    """
    df = df_in.copy()

    # ensure required inputs exist
    if "client_ip" not in df.columns:
        df["client_ip"] = ""
    if "dest_ip" not in df.columns:
        df["dest_ip"] = ""
    if "timestamp" not in df.columns:
        df["timestamp"] = ""

    if raw_col is None:
        raw_col = _pick_raw_col(df)
    if not raw_col:
        df["raw_log"] = ""
        raw_col = "raw_log"

    raw = df[raw_col].fillna("").astype(str)
    cip = df["client_ip"].fillna("").astype(str).str.strip()
    dip = df["dest_ip"].fillna("").astype(str).str.strip()

    # always define outputs (no missingness)
    df["ipv6_present_raw"] = np.int8(0)
    df["raw_ipv6_count"] = np.int16(0)
    df["raw_ipv6_first"] = ""

    df["dns_answer_ip"] = ""
    df["dns_answer_is_ipv6"] = np.int8(0)
    df["aaaa_query"] = raw.str.contains(_DNS_QTYPE_AAAA_RE, na=False).astype(np.int8)

    df["client_ipv6_unmasked_flag"] = np.int8(0)
    df["client_ipv6_unmasked"] = ""
    df["client_ipv6_type_id"] = np.int8(0)

    df["dest_ipv6_unmasked_flag"] = np.int8(0)
    df["dest_ipv6_unmasked"] = ""
    df["dest_ipv6_type_id"] = np.int8(0)

    df["tunnel_6to4_flag"] = np.int8(0)
    df["tunnel_teredo_flag"] = np.int8(0)
    df["tunnel_isatap_flag"] = np.int8(0)
    df["tunnel_nat64_flag"] = np.int8(0)
    df["tunnel_v4mapped_flag"] = np.int8(0)
    df["tunnel_ula_flag"] = np.int8(0)
    df["tunnel_linklocal_flag"] = np.int8(0)

    df["tunnel_proto41_hint"] = raw.str.contains(_PROTO41_RE, na=False).astype(np.int8)
    df["tunnel_udp3544_hint"] = raw.str.contains(_UDP3544_RE, na=False).astype(np.int8)
    df["tunnel_keywords_hint"] = raw.str.contains(_TUNNEL_KEYWORDS_RE, na=False).astype(np.int8)

    # DNS answer parse
    ans_tok = raw.str.extract(_DNS_ANSWER_RE, expand=False).fillna("")
    ans_ip, ans_ver = _normalize_ip_token_series(ans_tok)
    df["dns_answer_ip"] = ans_ip
    df["dns_answer_is_ipv6"] = (ans_ver == 6).astype(np.int8)

    # scan only plausible rows
    ipv6_candidate = raw.str.contains(_IPV6_HINT_RE, na=False) | (df["dns_answer_is_ipv6"] == 1)
    if ipv6_candidate.any():
        idxs = np.flatnonzero(ipv6_candidate.to_numpy())

        for j in idxs:
            rline = raw.iat[j]
            _v4s, v6s = _extract_valid_ips_from_raw(rline)

            # include DNS answer if IPv6
            if int(df["dns_answer_is_ipv6"].iat[j]) == 1:
                v6dns = df["dns_answer_ip"].iat[j]
                if v6dns and v6dns not in v6s:
                    v6s.append(v6dns)

            if not v6s:
                continue

            df.at[df.index[j], "ipv6_present_raw"] = np.int8(1)
            df.at[df.index[j], "raw_ipv6_count"] = np.int16(len(v6s))
            df.at[df.index[j], "raw_ipv6_first"] = v6s[0]

            has_6to4 = has_ter = has_isatap = has_nat64 = has_v4m = has_ula = has_ll = False

            for v6s_i in v6s:
                ip6 = _try_ip(v6s_i)
                if ip6 is None or getattr(ip6, "version", 0) != 6:
                    continue

                try:
                    if ip6 in NET_6TO4:
                        has_6to4 = True
                    if ip6 in NET_TEREDO:
                        has_ter = True
                    if ip6 in NAT64_WKP or ip6 in NAT64_LOCAL:
                        has_nat64 = True
                    if ip6 in NET_ULA:
                        has_ula = True
                    if ip6 in NET_LL:
                        has_ll = True
                except Exception:
                    pass

                _emb, typ = _ipv6_embedded_v4_and_type(ip6)  # type: ignore[arg-type]
                if typ == "v4mapped":
                    has_v4m = True
                if typ == "isatap":
                    has_isatap = True
                if typ == "teredo":
                    has_ter = True
                if typ == "6to4":
                    has_6to4 = True
                if typ == "nat64":
                    has_nat64 = True

            df.at[df.index[j], "tunnel_6to4_flag"] = np.int8(has_6to4)
            df.at[df.index[j], "tunnel_teredo_flag"] = np.int8(has_ter)
            df.at[df.index[j], "tunnel_isatap_flag"] = np.int8(has_isatap)
            df.at[df.index[j], "tunnel_nat64_flag"] = np.int8(has_nat64)
            df.at[df.index[j], "tunnel_v4mapped_flag"] = np.int8(has_v4m)
            df.at[df.index[j], "tunnel_ula_flag"] = np.int8(has_ula)
            df.at[df.index[j], "tunnel_linklocal_flag"] = np.int8(has_ll)

            # unmask: if parsed IPv4 is embedded in any IPv6 token
            cip4 = cip.iat[j]
            if cip4 and (":" not in cip4) and (cip4.count(".") == 3):
                v6, typ_id = _unmask_ipv6_from_ipv4(cip4, v6s)
                if v6:
                    df.at[df.index[j], "client_ipv6_unmasked_flag"] = np.int8(1)
                    df.at[df.index[j], "client_ipv6_unmasked"] = v6
                    df.at[df.index[j], "client_ipv6_type_id"] = np.int8(typ_id)

            dip4 = dip.iat[j]
            if dip4 and (":" not in dip4) and (dip4.count(".") == 3):
                v6, typ_id = _unmask_ipv6_from_ipv4(dip4, v6s)
                if v6:
                    df.at[df.index[j], "dest_ipv6_unmasked_flag"] = np.int8(1)
                    df.at[df.index[j], "dest_ipv6_unmasked"] = v6
                    df.at[df.index[j], "dest_ipv6_type_id"] = np.int8(typ_id)

    # ---------------------------------------------------------
    # Per-IP behavioral IPv6 metrics (needs timestamp)
    # ---------------------------------------------------------
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["_ts_utc"] = ts

    # If timestamps are missing, these group features become safe defaults (0)
    ipv6_present = pd.to_numeric(df["ipv6_present_raw"], errors="coerce").fillna(0).astype(np.int8)

    # ip_ipv6_event_rate
    try:
        df["ip_ipv6_event_rate"] = ipv6_present.groupby(cip, sort=False).transform("mean").astype(np.float32)
    except Exception:
        df["ip_ipv6_event_rate"] = np.float32(0.0)

    # hour bucket
    hour_bucket = ts.dt.floor("h")
    # v6 burst per ip per hour
    try:
        v6_per_ip_hour = ipv6_present.groupby([cip, hour_bucket], sort=False).transform("sum").astype(np.float32)
        mu_v6 = float(v6_per_ip_hour.mean()) if len(v6_per_ip_hour) else 0.0
        sd_v6 = float(v6_per_ip_hour.std()) if len(v6_per_ip_hour) else 0.0
        if not np.isfinite(sd_v6) or sd_v6 == 0.0:
            z_v6 = (v6_per_ip_hour - mu_v6)
        else:
            z_v6 = (v6_per_ip_hour - mu_v6) / sd_v6
        df["ip_ipv6_burst_hour"] = (z_v6 > 3.0).astype(np.int8)
    except Exception:
        df["ip_ipv6_burst_hour"] = np.int8(0)

    # aaaa burst
    try:
        aaaa = pd.to_numeric(df["aaaa_query"], errors="coerce").fillna(0).astype(np.int8)
        aaaa_per_ip_hour = aaaa.groupby([cip, hour_bucket], sort=False).transform("sum").astype(np.float32)
        mu_a = float(aaaa_per_ip_hour.mean()) if len(aaaa_per_ip_hour) else 0.0
        sd_a = float(aaaa_per_ip_hour.std()) if len(aaaa_per_ip_hour) else 0.0
        if not np.isfinite(sd_a) or sd_a == 0.0:
            z_a = (aaaa_per_ip_hour - mu_a)
        else:
            z_a = (aaaa_per_ip_hour - mu_a) / sd_a
        df["ip_aaaa_burst_hour"] = (z_a > 3.0).astype(np.int8)
    except Exception:
        df["ip_aaaa_burst_hour"] = np.int8(0)

    # switch rate (requires sorting by time)
    df["ip_ipv6_switch_rate"] = np.float32(0.0)
    try:
        tmp = df[["_ts_utc"]].copy()
        tmp["_idx"] = df.index
        tmp["client_ip"] = cip.values
        tmp["ipv6_present_raw"] = ipv6_present.values
        tmp = tmp.dropna(subset=["client_ip"])
        tmp.sort_values(["client_ip", "_ts_utc"], inplace=True, kind="mergesort")
        prev = tmp.groupby("client_ip", sort=False)["ipv6_present_raw"].shift(1)
        sw = (prev.notna() & (tmp["ipv6_present_raw"] != prev)).astype(np.int8)
        sw_count = sw.groupby(tmp["client_ip"], sort=False).transform("sum").astype(np.float32)
        ev_count = tmp.groupby("client_ip", sort=False)["ipv6_present_raw"].transform("size").astype(np.float32)
        rate = (sw_count / (ev_count + 1.0)).astype(np.float32)
        df.loc[tmp["_idx"].to_numpy(), "ip_ipv6_switch_rate"] = rate.to_numpy()
    except Exception:
        pass

    # ---------------------------------------------------------
    # Tunnel score (uses optional other columns if present)
    # ---------------------------------------------------------
    ua_release_anom = _safe_int_col(df, "ua_release_anomaly_flag", 0).astype(np.int8)
    high_bytes_out = _safe_int_col(df, "high_bytes_out", 0).astype(np.int8)
    threat_level_int = _safe_int_col(df, "threat_level_int", 0).astype(np.int16)
    command_risk_score = _safe_float_col(df, "command_risk_score", 0.0)
    session_multi_device_risk = _safe_float_col(df, "session_multi_device_risk", 0.0)

    base_score = (
        0.25 * df["tunnel_teredo_flag"].astype(float)
        + 0.25 * df["tunnel_6to4_flag"].astype(float)
        + 0.20 * df["tunnel_isatap_flag"].astype(float)
        + 0.20 * df["tunnel_proto41_hint"].astype(float)
        + 0.15 * df["tunnel_nat64_flag"].astype(float)
        + 0.10 * df["tunnel_udp3544_hint"].astype(float)
        + 0.10 * df["tunnel_keywords_hint"].astype(float)
        + 0.10 * df["ip_ipv6_burst_hour"].astype(float)
        + 0.10 * df["ip_aaaa_burst_hour"].astype(float)
    )

    score = base_score.copy()
    score += 0.10 * ((df["ipv6_present_raw"].astype(int) == 1) & (ua_release_anom > 0)).astype(float)
    score += 0.10 * ((df["ipv6_present_raw"].astype(int) == 1) & (high_bytes_out > 0)).astype(float)
    score += 0.05 * (command_risk_score >= 0.8).astype(float)
    score += 0.05 * (session_multi_device_risk >= 0.8).astype(float)

    df["ipv6_tunnel_score"] = score.clip(0.0, 1.0).astype(np.float32)

    df["ipv6_tunnel_level_int"] = np.select(
        [
            df["ipv6_tunnel_score"] >= float(score_level3_threshold),
            df["ipv6_tunnel_score"] >= float(score_level2_threshold),
            df["ipv6_tunnel_score"] >= float(score_level1_threshold),
        ],
        [3, 2, 1],
        default=0,
    ).astype(np.int8)

    df["tunnel_exfil_interaction"] = (
        (df["ipv6_tunnel_level_int"].astype(int) >= 2) & (threat_level_int.astype(int) >= 2)
    ).astype(np.int8)

    df["ipv6_primary_flag"] = (df["ipv6_tunnel_level_int"].astype(int) >= int(primary_level_threshold)).astype(np.int8)

    # cleanup helper column
    df.drop(columns=["_ts_utc"], inplace=True, errors="ignore")
    return df


__all__ = ["add_ipv6_primary_conditions"]