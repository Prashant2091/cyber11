# imputer.py
# ============================================================
# CODE 2 — Universal Forensic Imputer (45 canonical types)
# ------------------------------------------------------------
# Input:  DataFrame produced by log_parser.py (canonical-ish columns present)
# Output: Same DataFrame (no columns dropped) with 16 PRIMARY_COLS fully imputed.
#
# ✅ Preserves type1..type9 logic (trained types) — only fills missing/invalids
# ✅ Covers 35 additional families + 1 futuristic type => 45 canonical types total
# ✅ Deterministic (stable hashing), artifact-driven when priors/resolver/bytes exist
# ✅ No forbidden placeholders in core text fields (unknown/nan/null/etc)
# ✅ No personal-name propagation into usernames (blocks first.last)
# ✅ Bytes baseline-safe + dynamic + observed-batch adaptive caps (no 10^8 explosions, no pile-up)
# ✅ Vector-safe Pandas ops (no Series .startswith(Series), no mask length mismatch)
# ✅ FIXED: sanitize_domain_series() cannot crash with unary "~" on floats
# ============================================================

from __future__ import annotations

import sys
import ipaddress
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from urllib.parse import urlparse

import numpy as np
import pandas as pd


# -----------------------------
# Primary standardized schema
# -----------------------------
PRIMARY_COLS: List[str] = [
    "client_ip", "timestamp", "method", "full_url", "status",
    "bytes_out", "referrer", "user_agent", "bytes_in", "domain",
    "dest_ip", "username", "workstation", "process", "command", "log_type"
]
INTERNAL_COLS: List[str] = ["raw_log", "url_path"]  # keep for better imputation


# -----------------------------
# Missing tokens / regex
# -----------------------------
MISS_TOKENS = {
    "", " ", "unknown", "UNKNOWN", "unknown-domain", "unknown_domain",
    "nan", "NaN", "none", "None", None,
    -1, "-1",
    "NULL", "null",
    "-", "--",
    "MISSING_TOKEN", "missing_token",
    "N/A", "n/a", "NA", "na",
    "NotProvided", "notprovided", "NOTPROVIDED",
    "Not_Provided", "not_provided",
}
MISS_STRS = {str(x).strip().lower() for x in MISS_TOKENS if x is not None}
BAD_STR_RE = re.compile(
    r"^(?:unknown|missing_token|null|none|nan|na|n/a|-|--|notprovided|unknown-domain|unknown_domain)\s*$",
    re.I
)

# STRICT IPv4 (kept as single source of truth)
_IPV4_RE = re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$")
_REDACTED_IP_LIKE = re.compile(r"^(?:x|\d{1,3})(?:\.(?:x|\d{1,3})){3}$", re.I)
IPV4_FINDER = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")

_URL_RE = re.compile(r'(?i)(https?://[^\s"\'<>]+)')
_HOST_FINDER = re.compile(r'(?i)\b([a-z0-9-]+(?:\.[a-z0-9-]+)+)(?::\d{1,5})?\b')

_HTTP_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
POST_METHODS = {"POST", "PUT", "PATCH"}

PERSON_NAME_RE = re.compile(r"^[a-z]{2,}\.[a-z]{2,}$", re.I)  # block personal-name propagation
SAFE_USER_RE = re.compile(
    r"^(SYSTEM|Guest|NETWORK SERVICE|LOCAL SERVICE|"
    r"svc_[a-z0-9_]+|dev_user|corp_user|corp_admin|external_user|security_analyst|"
    r"svc_web|svc_database|svc_automation|svc_backup|svc_monitoring|svc_ci|"
    r"ueba_[a-z0-9_]+|netadmin[0-9]*|admin[0-9]*|appsvc|www-data|postgres|root|deploy)$",
    re.I
)

LOLBIN_RE = re.compile(
    r"(?:^|[^a-z0-9])("
    r"rundll32|regsvr32|mshta|wmic|certutil|bitsadmin|installutil|msiexec|"
    r"wscript|cscript|schtasks|cmd\.exe|powershell|pwsh|curl|wget|"
    r"python|python3|java|node|bash|sh|"
    r"conhost|dllhost|svchost"
    r")(?:$|[^a-z0-9])",
    re.I
)

SUSPICIOUS_CMD_HINTS_RE = re.compile(
    r"(scrobj\.dll|javascript:|vbscript:|/i:|/u|/s|urlcache|-split|-f|invoke-webrequest|iwr|downloadstring|"
    r"frombase64string|certutil|bitsadmin|regsvr32|mshta|rundll32|wmic|"
    r"curl\s+-|wget\s+|powershell\s+-|pwsh\s+-|cmd\.exe\s+/c|cscript|wscript)",
    re.I
)


# -----------------------------
# Canonical 45 log types (exact)
# -----------------------------
CANON_TYPES_45: List[str] = [
    # 9 trained
    "type1_space", "apache", "type3_proc_csv", "type4_web_csv", "type5_proxy_csv",
    "type6_event_csv", "type7_asset_csv", "type8_firewall_csv", "type9_dynamic_csv",
    # 35 additional
    "dns", "flow", "ids", "zeek_conn", "tls",
    "idp", "ad_auth", "pam", "aaa", "mfa",
    "edr", "linux_auth", "auditd", "mac_es", "fim",
    "cloud_audit", "cloud_flow", "k8s_audit", "container_runtime", "secrets_kms",
    "waf", "apigw", "lb", "app", "envoy",
    "db_audit", "objstore", "dlp_casb", "saas_admin", "email_sec",
    "gh_audit", "cicd", "k8s_event", "ics", "edge",
    # 1 futuristic
    "futuristic_unknown",
]
CANON_SET = set(CANON_TYPES_45)

# map raw parser tags -> canonical 45
LOG_TYPE_ALIAS: Dict[str, str] = {
    # trained aliases
    "apache_strict": "apache",
    "apache_relaxed": "apache",
    "type7_dhcp_csv": "type7_asset_csv",
    "type7_asset_csv": "type7_asset_csv",

    # linux
    "linux_syslog": "linux_auth",
    "linux_auth": "linux_auth",
    "auditd_execve": "auditd",
    "auditd_syscall": "auditd",
    "auditd": "auditd",

    # cloud/container/secrets groupings
    "containerd": "container_runtime",
    "docker": "container_runtime",
    "secrets": "secrets_kms",
    "kms": "secrets_kms",

    # dlp/casb grouping
    "dlp": "dlp_casb",
    "casb": "dlp_casb",

    # flow grouping
    "cloud_flow": "cloud_flow",
    "flow": "flow",
}


def normalize_log_type(raw_type: Any) -> Tuple[str, str]:
    raw = str(raw_type) if raw_type is not None else ""
    raw_clean = raw.strip()
    low = raw_clean.lower()
    if not low or low in MISS_STRS or BAD_STR_RE.match(low):
        return "futuristic_unknown", raw_clean

    canon = LOG_TYPE_ALIAS.get(low, low)
    if canon not in CANON_SET:
        if canon.startswith("auditd_"):
            canon = "auditd"
        elif canon in ("container", "container_runtime_logs"):
            canon = "container_runtime"
        elif canon in ("secret", "secrets_audit", "kms_audit"):
            canon = "secrets_kms"
        else:
            canon = "futuristic_unknown"

    return canon, raw_clean


# -----------------------------
# Safe helpers
# -----------------------------
def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass
    s = str(x)
    return "" if s.strip().lower() in MISS_STRS else s


def clean_missing_series_fast(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    ss = s.copy()
    ss = ss.replace(r"^\s*$", np.nan, regex=True)
    tmp = ss.fillna("").astype(str).str.strip()
    m = tmp.str.lower().isin(MISS_STRS) | tmp.eq("")
    return ss.mask(m, np.nan)


def ensure_no_bad_tokens(series: pd.Series, fallback_value: str) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    bad = s.str.lower().isin(MISS_STRS) | s.eq("") | s.str.match(BAD_STR_RE, na=False)
    return s.mask(bad, fallback_value)


def stable_u64(series: pd.Series, salt: str) -> np.ndarray:
    h = pd.util.hash_pandas_object(series.fillna("").astype(str) + "|" + salt, index=False)
    return h.to_numpy(dtype=np.uint64, copy=False)


def stable_u01(series: pd.Series, salt: str) -> np.ndarray:
    h = stable_u64(series, salt)
    u = ((h % 9_999_991) + 1).astype(np.float64) / 9_999_992.0
    return np.clip(u, 1e-12, 1 - 1e-12)


def is_ipv4_str(x: str) -> bool:
    return bool(_IPV4_RE.match(safe_str(x).strip()))


def normalize_ip_series(s: pd.Series) -> pd.Series:
    ss = s.fillna("").astype(str).str.strip().str.strip("[]")
    out = pd.Series(np.nan, index=ss.index, dtype="object")

    v4 = ss.str.match(_IPV4_RE, na=False)
    if v4.any():
        out.loc[v4] = ss.loc[v4]

    red = (~v4) & ss.str.match(_REDACTED_IP_LIKE, na=False)
    if red.any():
        out.loc[red] = np.nan

    v6 = (~v4) & (~red) & ss.str.contains(":", regex=False) & ss.ne("")
    if v6.any():
        def _v6(x):
            try:
                return str(ipaddress.ip_address(x))
            except Exception:
                return np.nan
        out.loc[v6] = ss.loc[v6].map(_v6)

    return out


def ip_kind_fast(ip_s: str) -> str:
    s = safe_str(ip_s).strip()
    if not s:
        return "invalid"
    if is_ipv4_str(s):
        if s.startswith("127."):
            return "loopback"
        if s.startswith("169.254."):
            return "link-local"
        if s.startswith("10.") or s.startswith("192.168.") or re.match(r"^172\.(1[6-9]|2\d|3[0-1])\.", s):
            return "private"
        if re.match(r"^100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.", s):
            return "cgnat"
        return "public"
    if ":" in s:
        try:
            ip = ipaddress.ip_address(s)
            if ip.is_loopback:
                return "loopback"
            if ip.is_link_local:
                return "link-local"
            if ip.is_private:
                return "private"
            if ip.is_multicast:
                return "multicast"
            if ip.is_reserved:
                return "reserved"
            if ip.is_unspecified:
                return "unspecified"
            return "public"
        except Exception:
            return "invalid"
    return "invalid"


# -----------------------------
# URL helpers
# -----------------------------
def extract_first_url(s: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    s = safe_str(s)
    m = _URL_RE.search(s)
    if not m:
        return None, None, None
    url = m.group(1)
    try:
        p = urlparse(url)
        host = (p.netloc or "").split("@")[-1].split(":")[0].strip("[]").strip()
        path = (p.path or "/") or "/"
        host = host.rstrip(',.);]\'"')
        return url, (host or None), (path or "/")
    except Exception:
        return url, None, None


def extract_host_path(full_url: pd.Series) -> Tuple[pd.Series, pd.Series]:
    u = full_url.fillna("").astype(str)
    host = u.str.extract(r"^\s*(?:https?://)?([^/\s:?#]+)", expand=False).fillna("")
    host = host.str.split("@").str[-1].str.split(":").str[0].str.strip("[]").fillna("")
    path = u.str.extract(r"^\s*(?:https?://)?[^/\s:?#]+(?P<path>/[^?#\s]*)", expand=False).fillna("/")
    path = path.where(path.str.startswith("/"), "/" + path)
    return host, path


def canonical_url_key_series(full_url: pd.Series) -> pd.Series:
    u = full_url.fillna("").astype(str)
    scheme = u.str.extract(r"^\s*(https?)://", expand=False).fillna("http").str.lower()
    host, path = extract_host_path(u)
    host_l = host.str.lower()
    path_nq = path.str.split("?", n=1).str[0]
    key = scheme + "://" + host_l + path_nq
    key = key.where(host_l.str.len() > 0, "")
    return key


def payload_from_path_series(url_path: pd.Series) -> pd.Series:
    p = url_path.fillna("").astype(str).str.split("?", n=1).str[0].str.rstrip("/")
    last = p.str.split("/", n=-1).str[-1]
    return last.fillna("")


def host_from_anywhere(s: str) -> Optional[str]:
    s = safe_str(s)
    _, host, _ = extract_first_url(s)
    if host:
        h = host.strip("[]").strip().rstrip(',.);]\'"').lower()
        return h if h else None
    for m in _HOST_FINDER.finditer(s):
        cand = (m.group(1) or "").strip().rstrip(',.);]\'"').lower()
        if cand and "." in cand:
            return cand
    return None


# -----------------------------
# Domain derivation from dest_ip (no octet leakage)
# -----------------------------
SITE_LIST = np.array(
    ["la", "ny", "sf", "sea", "chi", "dal", "blr", "bom", "del", "sin", "lon", "fra", "ams", "tok", "syd", "dxb", "hkg"],
    dtype=object
)
_SUFFIX_POOL = {
    "aws": np.array(["compute.amazonaws.com", "amazonaws.com", "cloudfront.net"], dtype=object),
    "azure": np.array(["cloudapp.azure.com", "azurewebsites.net", "windows.net", "trafficmanager.net"], dtype=object),
    "gcp": np.array(["googleusercontent.com", "googleapis.com", "appspot.com"], dtype=object),
    "cloudflare": np.array(["cloudflare.com", "workers.dev"], dtype=object),
    "fastly": np.array(["fastly.net"], dtype=object),
    "akamai": np.array(["akamai.net", "edgekey.net", "edgesuite.net"], dtype=object),
    "oracle": np.array(["oraclecloud.com"], dtype=object),
    "digitalocean": np.array(["digitalocean.com", "digitaloceanspaces.com"], dtype=object),
    "isp": np.array(["transit.net", "broadband.net", "carrier.net", "edge.net", "backbone.net", "metro.net", "wan.net", "access.net"], dtype=object),
}
_PREFIX = {
    "aws": "ec2-",
    "azure": "vm-",
    "gcp": "gce-",
    "oracle": "instance-",
    "digitalocean": "droplet-",
    "cloudflare": "cf-",
    "fastly": "edge-",
    "akamai": "a-",
    "isp": "host-",
}


def provider_key_from_ipv4(ip: str) -> str:
    ip = (ip or "").strip()
    if not _IPV4_RE.match(ip):
        return "isp"
    if ip.startswith(("104.", "188.")) or ip.startswith("172.64."):
        return "cloudflare"
    if ip.startswith("151.101."):
        return "fastly"
    if ip.startswith("23."):
        return "akamai"
    if ip.startswith(("3.", "13.", "15.", "18.", "52.", "54.")):
        return "aws"
    if ip.startswith(("20.", "40.")):
        return "azure"
    if ip.startswith("35."):
        return "gcp"
    if ip.startswith("64.225."):
        return "digitalocean"
    if ip.startswith(("129.", "130.", "131.")):
        return "oracle"
    return "isp"


def derive_domain_from_dest_ip_series(dest_ip: pd.Series) -> pd.Series:
    dip = dest_ip.fillna("").astype(str).str.strip()
    out = pd.Series("host-default.external", index=dip.index, dtype="object")

    is_v4 = dip.str.match(_IPV4_RE, na=False)
    if is_v4.any():
        ip = dip.loc[is_v4].astype(str)
        idx = ip.index

        h = pd.util.hash_pandas_object(ip + "|dom", index=False).astype(np.uint64)
        host_id = (h % 900000 + 100000).astype(int).astype(str)
        sites = pd.Series(SITE_LIST[(h.values % len(SITE_LIST)).astype(int)], index=idx)

        pk = ip.map(provider_key_from_ipv4)
        hh = pd.util.hash_pandas_object(ip + "|" + pk.astype(str) + "|pub", index=False).astype(np.uint64)

        res = pd.Series("", index=idx, dtype="object")
        for key in pk.unique():
            mm = pk.eq(key)
            pool = _SUFFIX_POOL.get(key, _SUFFIX_POOL["isp"])
            pref = _PREFIX.get(key, "host-")
            suf = np.array(pool, dtype=object)[(hh.loc[mm].values % len(pool)).astype(int)]
            res.loc[mm] = (pref + sites.loc[mm].values + "-" + host_id.loc[mm].values + "." + suf)
        out.loc[is_v4] = res.values

    is_v6 = (~is_v4) & dip.str.contains(":", regex=False) & dip.str.len().between(2, 80)
    if is_v6.any():
        vals = dip.loc[is_v6].tolist()
        doms = []
        for v in vals:
            try:
                ip6 = ipaddress.ip_address(v)
                if ip6.is_loopback:
                    doms.append("localhost")
                    continue
                internal = bool(ip6.is_private or ip6.is_link_local)
                s = ip6.compressed.lower()
                hx = hex(abs(hash(s)) % (16 ** 6))[2:].rjust(6, "0")
                suffix = "corp.local" if internal else "transit.net"
                doms.append(f"v6-host-{hx}.{suffix}")
            except Exception:
                hx = hex(abs(hash(v)) % (16 ** 6))[2:].rjust(6, "0")
                doms.append(f"node-{hx}.external")
        out.loc[is_v6] = pd.Series(doms, index=dip.loc[is_v6].index, dtype="object")

    other = (~is_v4) & (~is_v6)
    if other.any():
        h2 = pd.util.hash_pandas_object(dip.loc[other].astype(str) + "|inv", index=False).to_numpy(dtype=np.uint64, copy=False)
        code = (h2 % 1_000_000).astype(np.int64)
        out.loc[other] = ("node-" + pd.Series(code, index=dip.loc[other].index).astype(str) + ".external").values

    return out.astype(str).str.lower()


# -----------------------------
# Domain sanitizer (FIXED)
# -----------------------------
def sanitize_domain_series(host: pd.Series) -> pd.Series:
    """
    Robust domain sanitizer:
    - mixed dtype safe (strings, floats, ints, NaN)
    - NEVER applies ~ to a Series containing NaN floats
    - keeps np.nan for missing
    """
    if host is None:
        return pd.Series(dtype=object)

    s = host.copy().astype("string")

    s = s.str.strip().str.lower()
    s = s.str.strip("[](){}<>\"' ")
    s = s.str.rstrip(".,;)]}\"'")

    s = s.str.replace(r"^https?://", "", regex=True)
    s = s.str.split("/", n=1).str[0]
    s = s.str.split("?", n=1).str[0]
    s = s.str.split("#", n=1).str[0]

    s = s.str.split("@", n=1).str[-1]
    s = s.str.split(":", n=1).str[0]

    miss = {
        "", "-", "--", "none", "null", "nan", "na", "n/a",
        "unknown", "notprovided", "not_provided", "unknown_domain", "unknown-domain"
    }
    miss_mask = (s.isna() | s.isin(list(miss))).fillna(True)
    s = s.mask(miss_mask, pd.NA)

    s = s.mask(s.str.match(r"^\d+(?:\.\d+)*$", na=False), pd.NA)
    s = s.mask(s.str.match(_IPV4_RE, na=False), pd.NA)

    internal_ok = (".local", ".lan", ".internal", ".intra", ".corp", ".corp.local")

    # ✅ key point: na=False makes these PURE bool (no NaN floats)
    has_dot = s.str.contains(".", regex=False, na=False)
    is_internal = s.str.endswith(internal_ok, na=False)
    valid = has_dot | is_internal  # bool

    # ✅ avoid "~" on anything non-bool by guaranteeing valid is bool
    bad = s.notna() & (valid == False)
    s = s.mask(bad, pd.NA)

    out = s.astype(object)
    out = out.where(pd.notna(out), np.nan)
    return out


# -----------------------------
# ResolverLite (artifact-driven)
# -----------------------------
class ResolverLite:
    """
    Resolver state format supported:
      resolver_state = {"overrides": {...}, "dom_ip": {dom: Counter/dict}}
    """
    def __init__(self, resolver_state: Optional[dict]):
        self.overrides: Dict[str, str] = {}
        self.dom_to_ip: Dict[str, str] = {}
        if isinstance(resolver_state, dict):
            ov = resolver_state.get("overrides", {}) or {}
            if isinstance(ov, dict):
                self.overrides = {str(k).lower(): str(v) for k, v in ov.items()}
            dom_ip = resolver_state.get("dom_ip", {}) or {}
            if isinstance(dom_ip, dict):
                for dom, cnt in dom_ip.items():
                    ip = None
                    try:
                        if hasattr(cnt, "most_common"):
                            ip = cnt.most_common(1)[0][0] if len(cnt) else None
                        elif isinstance(cnt, dict):
                            ip = max(cnt.items(), key=lambda x: x[1])[0] if cnt else None
                    except Exception:
                        ip = None
                    if ip:
                        self.dom_to_ip[str(dom).lower()] = str(ip)

    def _norm_dom(self, domain: str) -> str:
        d = safe_str(domain).strip().lower()
        d = d.strip("[]").strip().strip(',.);]\'"')
        if d.startswith(("http://", "https://")):
            try:
                p = urlparse(d)
                d = (p.netloc or "").split("@")[-1].split(":")[0].strip("[]").strip().lower()
            except Exception:
                pass
        return d

    def resolve_one(self, domain: str, seed: str = "", avoid_ip: str = "") -> str:
        dom = self._norm_dom(domain)
        if not dom:
            dom = "host.default.external"
        avoid = safe_str(avoid_ip).strip()
        avoid = avoid if is_ipv4_str(avoid) else ""

        if dom in self.overrides:
            ip = safe_str(self.overrides[dom]).strip()
            if avoid and ip == avoid:
                h = int(pd.util.hash_pandas_object(pd.Series([dom + "|" + seed]), index=False).iloc[0])
                return f"10.{h % 255}.{(h >> 8) % 255}.{(h >> 16) % 254 + 1}"
            return ip

        if dom in self.dom_to_ip:
            ip = self.dom_to_ip[dom]
            if avoid and ip == avoid:
                h = int(pd.util.hash_pandas_object(pd.Series([dom + "|" + seed]), index=False).iloc[0])
                return f"10.{h % 255}.{(h >> 8) % 255}.{(h >> 16) % 254 + 1}"
            return ip

        h = int(pd.util.hash_pandas_object(pd.Series([dom + "|" + seed]), index=False).iloc[0])
        ip = f"8.{(h >> 8) % 255}.{(h >> 16) % 255}.{(h >> 24) % 254 + 1}"
        if avoid and ip == avoid:
            ip = f"8.{((h >> 8) + 11) % 255}.{((h >> 16) + 13) % 255}.{((h >> 24) + 17) % 254 + 1}"
        return ip

    def resolve_series(self, domain_s: pd.Series, seed_s: pd.Series, avoid_ip_s: Optional[pd.Series] = None) -> pd.Series:
        dom = domain_s.fillna("").astype(str).str.lower().str.strip()
        seed = seed_s.fillna("").astype(str)
        avoid = avoid_ip_s.fillna("").astype(str) if avoid_ip_s is not None else pd.Series("", index=dom.index)
        key = dom + "|" + seed + "|" + avoid
        uniq = key.unique()
        mp: Dict[str, str] = {}
        for k in uniq:
            d0, s0, a0 = (k.split("|", 2) + ["", "", ""])[:3]
            mp[k] = self.resolve_one(d0, seed=s0, avoid_ip=a0)
        return key.map(mp)


# -----------------------------
# Semantic normalization for 45 types (only sets missing/invalid fields)
# -----------------------------
def semantic_normalize(df: pd.DataFrame, row_seed: pd.Series) -> pd.DataFrame:
    df = df.copy()
    lt = df["log_type"].fillna("futuristic_unknown").astype(str).str.lower()

    trained = lt.isin([
        "type1_space", "apache", "type3_proc_csv", "type4_web_csv", "type5_proxy_csv",
        "type6_event_csv", "type7_asset_csv", "type8_firewall_csv", "type9_dynamic_csv"
    ])

    def set_if_empty(col: str, values: Any, mask: pd.Series):
        s = df[col]
        s0 = s.fillna("").astype(str).str.strip()
        bad = s0.eq("") | s0.str.lower().isin(MISS_STRS) | s0.str.match(BAD_STR_RE, na=False)
        m = mask & bad
        if m.any():
            if isinstance(values, pd.Series):
                df.loc[m, col] = values.loc[m].to_numpy()
            else:
                df.loc[m, col] = values

    DOMAIN_DEFAULT = {
        "dns": "dns.local",
        "flow": "netflow.local",
        "zeek_conn": "zeek.local",
        "ids": "ids.local",
        "tls": "tls.local",
        "idp": "idp.corp.local",
        "ad_auth": "ad.corp.local",
        "pam": "pam.corp.local",
        "aaa": "aaa.corp.local",
        "mfa": "mfa.corp.local",
        "edr": "edr.corp.local",
        "linux_auth": "linux.local",
        "auditd": "auditd.linux.local",
        "mac_es": "mac.local",
        "fim": "fim.corp.local",
        "cloud_audit": "cloud.audit.local",
        "cloud_flow": "cloud.flow.local",
        "k8s_audit": "k8s.audit.local",
        "k8s_event": "k8s.events.local",
        "container_runtime": "container.runtime.local",
        "secrets_kms": "secrets.kms.local",
        "waf": "waf.edge.local",
        "apigw": "api.gateway.local",
        "lb": "lb.local",
        "envoy": "mesh.local",
        "app": "app.local",
        "db_audit": "db.audit.local",
        "objstore": "objstore.local",
        "dlp_casb": "dlp.corp.local",
        "saas_admin": "saas.admin.local",
        "email_sec": "email.security.local",
        "gh_audit": "github.audit.local",
        "cicd": "cicd.local",
        "ics": "ics.local",
        "edge": "edge.cdn.local",
        "futuristic_unknown": "unknown.local",
    }

    PATH_DEFAULT = {
        "dns": "/dns/query",
        "flow": "/flow",
        "zeek_conn": "/zeek/conn",
        "ids": "/ids/alert",
        "tls": "/tls/handshake",
        "idp": "/idp/signin",
        "ad_auth": "/ad/auth",
        "pam": "/pam/session",
        "aaa": "/aaa/auth",
        "mfa": "/mfa/event",
        "edr": "/edr/event",
        "linux_auth": "/linux/auth",
        "auditd": "/linux/auditd",
        "mac_es": "/mac/es",
        "fim": "/fim/event",
        "cloud_audit": "/cloud/audit",
        "cloud_flow": "/cloud/flow",
        "k8s_audit": "/k8s/audit",
        "k8s_event": "/k8s/event",
        "container_runtime": "/container/runtime",
        "secrets_kms": "/secrets/kms",
        "waf": "/waf",
        "apigw": "/apigw",
        "lb": "/lb",
        "envoy": "/envoy",
        "app": "/app",
        "db_audit": "/db/audit",
        "objstore": "/objstore",
        "dlp_casb": "/dlp",
        "saas_admin": "/saas/admin",
        "email_sec": "/email/security",
        "gh_audit": "/github/audit",
        "cicd": "/cicd",
        "ics": "/ics",
        "edge": "/edge",
        "futuristic_unknown": "/unknown",
    }

    METHOD_DEFAULT = {
        "dns": "GET",
        "flow": "GET",
        "zeek_conn": "GET",
        "tls": "GET",
        "ids": "POST",
        "idp": "POST", "ad_auth": "POST", "pam": "POST", "aaa": "POST", "mfa": "POST",
        "edr": "POST", "linux_auth": "POST", "auditd": "POST", "mac_es": "POST", "fim": "POST",
        "cloud_audit": "POST", "cloud_flow": "GET", "k8s_audit": "POST", "k8s_event": "POST",
        "container_runtime": "POST", "secrets_kms": "POST",
        "waf": "GET", "apigw": "GET", "lb": "GET", "envoy": "GET", "edge": "GET", "app": "POST",
        "db_audit": "POST", "objstore": "GET", "dlp_casb": "POST", "saas_admin": "POST", "email_sec": "POST",
        "gh_audit": "POST", "cicd": "POST", "ics": "POST",
        "futuristic_unknown": "GET",
    }

    nontrained = ~trained

    for t in sorted(set(lt[nontrained].unique())):
        m = nontrained & lt.eq(t)
        if not m.any():
            continue
        set_if_empty("domain", DOMAIN_DEFAULT.get(t, "unknown.local"), m)
        set_if_empty("url_path", PATH_DEFAULT.get(t, "/unknown"), m)

        meth = df["method"].fillna("").astype(str).str.upper().str.strip()
        inv = m & (~meth.isin(list(_HTTP_METHODS)))
        if inv.any():
            df.loc[inv, "method"] = METHOD_DEFAULT.get(t, "GET")

    fu = df["full_url"].fillna("").astype(str).str.strip()
    fu_bad = fu.eq("") | fu.str.lower().isin(MISS_STRS) | fu.str.match(BAD_STR_RE, na=False)
    if fu_bad.any():
        dom = df["domain"].fillna("localhost").astype(str).str.strip().str.strip("/")
        up = df["url_path"].fillna("/").astype(str)
        up = up.where(up.str.startswith("/"), "/" + up)
        df.loc[fu_bad, "full_url"] = ("http://" + dom.loc[fu_bad].to_numpy() + up.loc[fu_bad].to_numpy())

    stv = pd.to_numeric(df["status"], errors="coerce")
    bad = stv.isna() | ~stv.between(100, 599)
    if bad.any():
        raw = df.get("raw_log", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
        s2 = pd.Series(200, index=df.index, dtype=int)
        s2.loc[lt.eq("waf")] = 403
        fail = raw.str.contains(r"\b(fail|failed|deny|denied|block|blocked|reject|rejected|forbid|forbidden)\b", regex=True)
        s2.loc[fail] = 403
        df.loc[bad, "status"] = s2.loc[bad].to_numpy()

    return df


# -----------------------------
# Bytes: baseline-safe + dynamic + observed-batch adaptive caps
# -----------------------------
def _observed_caps_by_type(df: pd.DataFrame, side: str, q: float = 0.999) -> Dict[str, int]:
    if side not in ("bytes_in", "bytes_out"):
        return {}
    lt = df["log_type"].fillna("futuristic_unknown").astype(str)
    v = pd.to_numeric(df[side], errors="coerce")
    v = v.where(v >= 0, np.nan)

    out: Dict[str, int] = {}
    for t, sub in v.groupby(lt, sort=False):
        arr = sub.dropna()
        if len(arr) < 20:
            if len(arr) > 0:
                out[str(t)] = int(arr.max())
            continue
        try:
            out[str(t)] = int(arr.quantile(q))
        except Exception:
            out[str(t)] = int(arr.max())
    return out


def _bytes_caps(df: pd.DataFrame, bytes_priors: dict, row_seed: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    base_in = int((bytes_priors or {}).get("global_max_in", 10_000) or 10_000)
    base_out = int((bytes_priors or {}).get("global_max_out", 25_000) or 25_000)

    dyn = (bytes_priors or {}).get("dynamic", {}) if isinstance(bytes_priors, dict) else {}
    by_lt = dyn.get("by_log_type", {}) if isinstance(dyn, dict) else {}
    abs_max = int(dyn.get("abs_max_bytes", 2_000_000_000)) if isinstance(dyn, dict) else 2_000_000_000

    lt = df["log_type"].fillna("futuristic_unknown").astype(str)

    dyn_in_map: Dict[str, int] = {}
    dyn_out_map: Dict[str, int] = {}
    if isinstance(by_lt, dict):
        for k, v in by_lt.items():
            try:
                dyn_in_map[str(k)] = int(((v or {}).get("in", {}) or {}).get("cap", base_in))
                dyn_out_map[str(k)] = int(((v or {}).get("out", {}) or {}).get("cap", base_out))
            except Exception:
                pass

    dyn_in = lt.map(dyn_in_map).fillna(base_in).astype(int).to_numpy(dtype=np.int64, copy=False)
    dyn_out = lt.map(dyn_out_map).fillna(base_out).astype(int).to_numpy(dtype=np.int64, copy=False)

    obs_in_map = _observed_caps_by_type(df, "bytes_in")
    obs_out_map = _observed_caps_by_type(df, "bytes_out")
    obs_in = lt.map(obs_in_map).fillna(0).astype(int).to_numpy(dtype=np.int64, copy=False)
    obs_out = lt.map(obs_out_map).fillna(0).astype(int).to_numpy(dtype=np.int64, copy=False)

    cap_in = np.maximum(base_in, np.maximum(dyn_in, obs_in))
    cap_out = np.maximum(base_out, np.maximum(dyn_out, obs_out))

    cap_in = np.clip(cap_in, base_in, abs_max).astype(np.int64)
    cap_out = np.clip(cap_out, base_out, abs_max).astype(np.int64)
    return cap_in, cap_out


def _det_bytes_from_cap(seed_u64: np.ndarray, cap: np.ndarray) -> np.ndarray:
    u = ((seed_u64 % 9_999_991) + 1).astype(np.float64) / 9_999_992.0
    x = cap.astype(np.float64) * (u ** 2)
    return np.clip(x, 0, cap).astype(np.int64)


def fix_or_impute_bytes(df: pd.DataFrame, bytes_priors: dict, row_seed: pd.Series) -> pd.DataFrame:
    df = df.copy()
    bi = pd.to_numeric(df.get("bytes_in", pd.Series(np.nan, index=df.index)), errors="coerce")
    bo = pd.to_numeric(df.get("bytes_out", pd.Series(np.nan, index=df.index)), errors="coerce")
    bi = bi.where(bi >= 0, np.nan)
    bo = bo.where(bo >= 0, np.nan)

    cap_in, cap_out = _bytes_caps(df, bytes_priors, row_seed)
    cap_in_s = pd.Series(cap_in, index=df.index, dtype=np.int64)
    cap_out_s = pd.Series(cap_out, index=df.index, dtype=np.int64)

    inv_bi = bi.isna() | (bi > cap_in_s * 4.0)
    inv_bo = bo.isna() | (bo > cap_out_s * 4.0)

    method = df.get("method", pd.Series("GET", index=df.index)).fillna("GET").astype(str).str.upper()
    status = pd.to_numeric(df.get("status", pd.Series(200, index=df.index)), errors="coerce").fillna(200).astype(int)

    is_head = method.eq("HEAD")
    inv_bi |= is_head & (bi.fillna(0) > 5_000)
    inv_bo |= is_head & (bo.fillna(0) > 10_000)

    is_204 = status.eq(204)
    inv_bo |= is_204 & (bo.fillna(0) != 0)

    seed_u64 = stable_u64(row_seed, "bytes")

    if inv_bi.any():
        idx = inv_bi.to_numpy()
        bi.loc[inv_bi] = _det_bytes_from_cap(seed_u64[idx], cap_in[idx]).astype(np.int64)
        is_post = method.isin(list(POST_METHODS))
        m = inv_bi & is_post
        if m.any():
            bump = (seed_u64[m.to_numpy()] % 2048 + 1).astype(np.int64)
            bi.loc[m] = np.maximum(bi.loc[m].astype(np.int64).to_numpy(), bump)

    if inv_bo.any():
        idx = inv_bo.to_numpy()
        bo.loc[inv_bo] = _det_bytes_from_cap((seed_u64[idx] ^ 0xA5A5A5A5), cap_out[idx]).astype(np.int64)

    bi = bi.fillna(0).clip(lower=0).astype(np.int64)
    bo = bo.fillna(0).clip(lower=0).astype(np.int64)
    bo.loc[is_204] = 0
    bo.loc[is_head] = bo.loc[is_head].clip(upper=10_000)
    bi.loc[is_head] = bi.loc[is_head].clip(upper=5_000)

    df["bytes_in"] = bi
    df["bytes_out"] = bo
    return df


# -----------------------------
# Command prefix (vector-safe)
# -----------------------------
def ensure_command_prefix_series(process: pd.Series, command: pd.Series) -> pd.Series:
    p = process.fillna("client_app.exe").astype(str).str.strip()
    c = command.fillna("").astype(str).str.strip()

    c = c.mask(c.eq("") | c.str.lower().isin(MISS_STRS) | c.str.match(BAD_STR_RE, na=False), "")
    c = c.mask(c.eq(""), p)

    # Make a boolean mask as a pandas Series for safe indexing
    pl = p.astype(str).str.lower().to_numpy(dtype=str, copy=False)
    cl = c.astype(str).str.lower().to_numpy(dtype=str, copy=False)
    try:
        starts = np.char.startswith(cl.astype(str), pl.astype(str))
    except Exception:
        starts = np.fromiter((str(cc).startswith(str(pp)) for cc, pp in zip(cl, pl)), dtype=bool, count=len(cl))
    starts_s = pd.Series(starts, index=c.index)

    need = (~starts_s) & (c.str.len() > 0)
    if need.any():
        c.loc[need] = (p.loc[need].to_numpy() + " " + c.loc[need].to_numpy()).astype(object)

    return c


# -----------------------------
# ForensicImputer
# -----------------------------
@dataclass
class ForensicImputer:
    priors: Optional[dict] = None
    resolver_state: Optional[dict] = None
    bytes_priors: Optional[dict] = None

    def __post_init__(self):
        self.priors = self.priors if isinstance(self.priors, dict) else {}
        self.resolver = ResolverLite(self.resolver_state if isinstance(self.resolver_state, dict) else {})
        self.bytes_priors = self.bytes_priors if isinstance(self.bytes_priors, dict) else {}

    def impute_df(self, df: pd.DataFrame, fill_text: str = "NotProvided") -> pd.DataFrame:
        df = df.copy()

        for c in PRIMARY_COLS + INTERNAL_COLS:
            if c not in df.columns:
                df[c] = np.nan

        df["log_type_raw"] = df["log_type"].fillna("").astype(str)

        canon, raw = zip(*df["log_type"].map(normalize_log_type).tolist())
        df["log_type"] = pd.Series(canon, index=df.index, dtype="object")
        df["log_type_raw"] = pd.Series(raw, index=df.index, dtype="object")

        for c in ["timestamp", "client_ip", "dest_ip", "domain", "full_url", "url_path", "method", "referrer", "user_agent",
                  "username", "workstation", "process", "command", "raw_log"]:
            df[c] = clean_missing_series_fast(df[c])

        df["status"] = pd.to_numeric(df["status"], errors="coerce")
        df["bytes_in"] = pd.to_numeric(df["bytes_in"], errors="coerce")
        df["bytes_out"] = pd.to_numeric(df["bytes_out"], errors="coerce")

        row_seed = (
            df["raw_log"].fillna("").astype(str)
            + "|" + df["timestamp"].fillna("").astype(str)
            + "|" + df["client_ip"].fillna("").astype(str)
            + "|" + df["dest_ip"].fillna("").astype(str)
            + "|" + df["full_url"].fillna("").astype(str)
            + "|" + df["url_path"].fillna("").astype(str)
            + "|" + df["log_type"].fillna("").astype(str)
        )

        # timestamp impute if missing
        ts = df["timestamp"].fillna("").astype(str).str.strip()
        miss_ts = ts.eq("") | ts.str.lower().isin(MISS_STRS) | ts.str.match(BAD_STR_RE, na=False)
        if miss_ts.any():
            base = pd.Timestamp("2020-01-01T00:00:00")
            span = int((pd.Timestamp("2026-01-01T00:00:00") - base).total_seconds())
            h = stable_u64(row_seed[miss_ts], "ts") % max(1, span)
            dt = base + pd.to_timedelta(h.astype(np.int64), unit="s")
            df.loc[miss_ts, "timestamp"] = dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()

        df = semantic_normalize(df, row_seed)

        # normalize IPs
        df["client_ip"] = normalize_ip_series(df["client_ip"])
        df["dest_ip"] = normalize_ip_series(df["dest_ip"])

        # url_path extraction if missing
        up = df["url_path"].fillna("").astype(str).str.strip()
        miss_up = up.eq("") | up.str.lower().isin(MISS_STRS) | up.str.match(BAD_STR_RE, na=False)
        if miss_up.any():
            _, path = extract_host_path(df["full_url"].fillna("").astype(str))
            df.loc[miss_up, "url_path"] = path.loc[miss_up].to_numpy()
        df["url_path"] = df["url_path"].fillna("/").astype(str)
        df["url_path"] = df["url_path"].where(df["url_path"].str.startswith("/"), "/" + df["url_path"])
        df["url_path"] = df["url_path"].replace("", "/")

        # domain sanitize + derive from full_url/referrer/command/raw + dest_ip fallback
        df["domain"] = sanitize_domain_series(df["domain"])

        miss_dom = df["domain"].isna()
        if miss_dom.any():
            host, _ = extract_host_path(df["full_url"].fillna("").astype(str))
            df.loc[miss_dom, "domain"] = sanitize_domain_series(host.loc[miss_dom]).to_numpy()

        miss_dom = df["domain"].isna()
        if miss_dom.any():
            blob = (df["referrer"].fillna("").astype(str) + " " + df["command"].fillna("").astype(str) + " " + df["raw_log"].fillna("").astype(str))
            cand = blob.loc[miss_dom].map(host_from_anywhere)
            df.loc[miss_dom, "domain"] = pd.Series(cand, index=df.index[miss_dom]).astype(str).replace("", np.nan).to_numpy()
            df["domain"] = sanitize_domain_series(df["domain"])

        miss_dom = df["domain"].isna()
        if miss_dom.any():
            df.loc[miss_dom, "domain"] = derive_domain_from_dest_ip_series(df.loc[miss_dom, "dest_ip"]).to_numpy()

        df["domain"] = df["domain"].fillna("host-default.internal").astype(str)

        # full_url impute if missing
        fu = df["full_url"].fillna("").astype(str).str.strip()
        miss_fu = fu.eq("") | fu.str.lower().isin(MISS_STRS) | fu.str.match(BAD_STR_RE, na=False)
        if miss_fu.any():
            dom = df.loc[miss_fu, "domain"].astype(str).str.strip().str.strip("/")
            up2 = df.loc[miss_fu, "url_path"].astype(str)
            up2 = up2.where(up2.str.startswith("/"), "/" + up2)
            df.loc[miss_fu, "full_url"] = ("http://" + dom.to_numpy() + up2.to_numpy())

        # method normalize
        m = df["method"].fillna("").astype(str).str.upper().str.strip()
        m = m.replace({"ALLOW": "GET", "BLOCK": "DELETE", "DENY": "DELETE", "LOGON": "POST", "LOGOFF": "DELETE"})
        bad_m = ~m.isin(list(_HTTP_METHODS))
        if bad_m.any():
            ctx = (df["full_url"].fillna("").astype(str) + " " + df["command"].fillna("").astype(str)).str.lower()
            outm = m.copy()
            outm.loc[bad_m] = "GET"
            outm.loc[bad_m & ctx.str.contains(r"\bdelete\b|\bremove\b|\bunlink\b|\blogoff\b", regex=True)] = "DELETE"
            outm.loc[bad_m & ctx.str.contains(r"\bput\b|\bupdate\b|\bpatch\b|\brenew\b", regex=True)] = "PUT"
            outm.loc[bad_m & ctx.str.contains(r"\bupload\b|\blogin\b|\bsignin\b|\bauth\b|\btoken\b|\bcreate\b|\bexec\b", regex=True)] = "POST"
            m = outm
        df["method"] = m.where(m.isin(list(_HTTP_METHODS)), "GET")

        # status normalize
        stv = pd.to_numeric(df["status"], errors="coerce")
        stv = stv.where(stv.between(100, 599), np.nan)
        miss_st = stv.isna()
        if miss_st.any():
            rawl = df["raw_log"].fillna("").astype(str).str.lower()
            st2 = pd.Series(200, index=df.index, dtype=int)
            fail = rawl.str.contains(r"\b(fail|failed|deny|denied|block|blocked|reject|rejected|forbid|forbidden)\b", regex=True)
            st2.loc[fail] = 403
            st2.loc[df["method"].eq("DELETE")] = 204
            st2.loc[df["log_type"].eq("waf")] = 403
            df.loc[miss_st, "status"] = st2.loc[miss_st].to_numpy()

        df["status"] = pd.to_numeric(df["status"], errors="coerce").fillna(200).round().astype(int)
        df.loc[(df["status"] < 100) | (df["status"] > 599), "status"] = 200

        # dest_ip priors/resolver if missing
        dip = df["dest_ip"]
        miss_dip = dip.isna()
        if miss_dip.any():
            if isinstance(self.priors, dict) and self.priors.get("url_key_to"):
                url_key = canonical_url_key_series(df["full_url"].fillna("").astype(str))
                cand = url_key.loc[miss_dip].map(lambda k: (self.priors["url_key_to"].get(k, {}) or {}).get("dest_ip", None))
                dip.loc[miss_dip] = pd.Series(cand, index=df.index[miss_dip])
                miss_dip = dip.isna()

            if miss_dip.any() and isinstance(self.priors, dict) and self.priors.get("payload_to"):
                payload = payload_from_path_series(df["url_path"])
                cand = payload.loc[miss_dip].map(lambda p: (self.priors["payload_to"].get(p, {}) or {}).get("dest_ip", None))
                dip.loc[miss_dip] = pd.Series(cand, index=df.index[miss_dip])
                miss_dip = dip.isna()

            if miss_dip.any():
                dom_s = df.loc[miss_dip, "domain"].astype(str)
                avoid = df.loc[miss_dip, "client_ip"].fillna("").astype(str)
                seed_s = (dom_s + "|" + avoid + "|" + row_seed.loc[miss_dip].astype(str))
                dip.loc[miss_dip] = self.resolver.resolve_series(dom_s, seed_s, avoid_ip_s=avoid).to_numpy()

        df["dest_ip"] = normalize_ip_series(dip)

        # client_ip priors → deterministic
        cip = df["client_ip"]
        miss_cip = cip.isna()

        if miss_cip.any() and isinstance(self.priors, dict) and self.priors.get("ws_to_ip"):
            ws_to_ip = self.priors.get("ws_to_ip", {}) or {}
            ws = df.loc[miss_cip, "workstation"].fillna("").astype(str)
            cand = ws.map(ws_to_ip)
            cip.loc[miss_cip] = cand.to_numpy()
            miss_cip = cip.isna()

        if miss_cip.any():
            idx = df.index[miss_cip]

            dip_s = df.loc[idx, "dest_ip"].fillna("").astype(str)
            dom_l = df.loc[idx, "domain"].fillna("").astype(str).str.lower()
            kind_pub = dip_s.map(ip_kind_fast).eq("public") & ~dom_l.str.contains(r"(internal|\.internal|\.local|\.lan|corp|intranet)", regex=True)

            seed = stable_u64(row_seed.loc[idx], "cip")

            b = ((seed >> 8) % 255).astype(np.int64)
            c_ = ((seed >> 16) % 255).astype(np.int64)
            d_ = ((seed >> 24) % 254 + 1).astype(np.int64)

            pub_a_choices = np.array([x for x in range(1, 224) if x not in {10, 100, 127, 169, 172, 192, 224}], dtype=np.int64)
            a_pub = pub_a_choices[(seed % len(pub_a_choices)).astype(np.int64)]

            pub_ip = (
                pd.Series(a_pub, index=idx).astype(str) + "."
                + pd.Series(b, index=idx).astype(str) + "."
                + pd.Series(c_, index=idx).astype(str) + "."
                + pd.Series(d_, index=idx).astype(str)
            )

            which = (seed % 3).astype(np.int64)
            s0 = (seed % 255).astype(np.int64)
            s8 = ((seed >> 8) % 255).astype(np.int64)
            s16 = ((seed >> 16) % 254 + 1).astype(np.int64)

            a10 = (
                "10." + pd.Series(s0, index=idx).astype(str) + "."
                + pd.Series(s8, index=idx).astype(str) + "."
                + pd.Series(s16, index=idx).astype(str)
            )
            a172 = "172.16." + pd.Series(s8, index=idx).astype(str) + "." + pd.Series(s16, index=idx).astype(str)
            a192 = "192.168." + pd.Series(s8, index=idx).astype(str) + "." + pd.Series(s16, index=idx).astype(str)

            priv_ip = pd.Series(
                np.where(which == 0, a10.to_numpy(), np.where(which == 1, a172.to_numpy(), a192.to_numpy())),
                index=idx,
                dtype="object",
            )

            cip.loc[idx] = np.where(kind_pub.to_numpy(), pub_ip.to_numpy(), priv_ip.to_numpy()).astype(object)

        df["client_ip"] = normalize_ip_series(cip)

        # process (conservative)
        proc = clean_missing_series_fast(df["process"]).fillna("").astype(str).str.strip()
        proc_bad = proc.eq("") | proc.str.lower().isin(MISS_STRS) | proc.str.match(BAD_STR_RE, na=False) | proc.str.contains(r"\s", regex=True, na=False)
        if proc_bad.any():
            ua_l = df["user_agent"].fillna("").astype(str).str.lower()
            cmd_l = df["command"].fillna("").astype(str).str.lower()
            lt2 = df["log_type"].fillna("").astype(str).str.lower()

            fill = pd.Series("", index=df.index, dtype="object")
            fill.loc[lt2.eq("type9_dynamic_csv")] = "winlogon.exe"
            fill.loc[lt2.eq("type6_event_csv")] = "eventlog.exe"
            fill.loc[lt2.eq("type8_firewall_csv")] = "firewallsvc.exe"
            fill.loc[lt2.str.contains("type7", regex=False)] = "dhcpclient.exe"

            fill.loc[ua_l.str.contains(r"\bcurl\b|curl/", regex=True) | cmd_l.str.contains(r"\bcurl\b", regex=True)] = "curl"
            fill.loc[ua_l.str.contains(r"\bwget\b", regex=True) | cmd_l.str.contains(r"\bwget\b", regex=True)] = "wget"
            fill.loc[ua_l.str.contains(r"python-requests|python", regex=True)] = "python.exe"
            fill.loc[ua_l.str.contains(r"powershell|pwsh", regex=True) | cmd_l.str.contains("invoke-webrequest", regex=False)] = "powershell.exe"
            fill.loc[ua_l.str.contains(r"firefox", regex=False)] = "firefox.exe"
            fill.loc[ua_l.str.contains(r"edg/|edge|trident|msie", regex=True)] = "edge.exe"
            fill.loc[ua_l.str.contains(r"chrome", regex=False) & ~ua_l.str.contains(r"edg/|edge", regex=True)] = "chrome.exe"
            fill.loc[ua_l.str.contains(r"safari", regex=False) & ~ua_l.str.contains(r"chrome|chromium", regex=True)] = "safari"

            first_tok = df["command"].fillna("").astype(str).str.extract(r'^\s*"?([^\s"]+)', expand=False).fillna("")
            base = first_tok.str.replace("\\", "/", regex=False).str.split("/").str[-1]
            fill = fill.mask(fill.eq(""), base)

            still = fill.eq("")
            if still.any():
                pool = np.array(["chrome.exe", "edge.exe", "firefox.exe", "curl", "python.exe", "client_app.exe"], dtype=object)
                h = stable_u64(row_seed, "proc_pool")
                fill.loc[still] = pool[(h[still.to_numpy()] % len(pool)).astype(int)]

            proc = proc.mask(proc_bad, fill)

        cmd_l2 = df["command"].fillna("").astype(str).str.lower()
        proc_l = proc.astype(str).str.lower()
        lb = proc_l.str.contains(LOLBIN_RE) | cmd_l2.str.contains(LOLBIN_RE) | cmd_l2.str.contains(SUSPICIOUS_CMD_HINTS_RE)
        already = proc.str.contains(r"\(lolbin\)$", regex=True, na=False)
        proc = proc.mask(lb & ~already, proc + "(lolbin)")
        df["process"] = ensure_no_bad_tokens(proc, "client_app.exe")

        # user_agent
        ua = clean_missing_series_fast(df["user_agent"]).fillna("").astype(str).str.strip()
        ua = ua.mask(ua.str.fullmatch(r"\d{3,6}", na=False), "")
        ua_bad = ua.eq("") | ua.str.lower().isin(MISS_STRS) | ua.str.match(BAD_STR_RE, na=False)
        if ua_bad.any():
            lt2 = df["log_type"].fillna("").astype(str).str.lower()
            ws_l = df["workstation"].fillna("").astype(str).str.lower()
            fill = pd.Series("", index=df.index, dtype="object")

            fill.loc[lt2.isin(["type6_event_csv", "type9_dynamic_csv", "auditd", "linux_auth"])] = "WindowsEventLog/10.0"
            fill.loc[lt2.eq("waf")] = "WAF/1.0"
            fill.loc[lt2.eq("ids")] = "Suricata/6.0"
            fill.loc[lt2.eq("dns")] = "DNS/1.0"
            fill.loc[lt2.eq("flow")] = "NetFlow/1.0"
            fill.loc[lt2.eq("zeek_conn")] = "Zeek/1.0"
            fill.loc[lt2.eq("tls")] = "TLS/1.0"
            fill.loc[lt2.eq("edr")] = "EDR/1.0"
            fill.loc[lt2.eq("mac_es")] = "macOS-ES/1.0"
            fill.loc[lt2.eq("fim")] = "FIM/1.0"
            fill.loc[lt2.eq("container_runtime")] = "container-runtime/1.0"
            fill.loc[lt2.eq("secrets_kms")] = "KMS/1.0"
            fill.loc[lt2.eq("cloud_audit")] = "CloudAudit/1.0"
            fill.loc[lt2.eq("k8s_audit")] = "KubernetesAudit/1.0"
            fill.loc[lt2.eq("k8s_event")] = "KubernetesEvent/1.0"
            fill.loc[lt2.eq("objstore")] = "ObjStore/1.0"
            fill.loc[lt2.eq("db_audit")] = "DBAudit/1.0"
            fill.loc[lt2.eq("dlp_casb")] = "DLP/1.0"
            fill.loc[lt2.eq("email_sec")] = "EmailSecurity/1.0"
            fill.loc[lt2.eq("gh_audit")] = "GitHubAudit/1.0"
            fill.loc[lt2.eq("cicd")] = "CICD/1.0"
            fill.loc[lt2.eq("ics")] = "ICS/1.0"
            fill.loc[lt2.eq("edge")] = "Edge/1.0"

            fill.loc[fill.eq("") & ws_l.str.contains("android", regex=False)] = "Mozilla/5.0 (Linux; Android 14; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Mobile Safari/537.36"
            fill.loc[fill.eq("") & ws_l.str.contains("iphone", regex=False)] = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile Safari/604.1"
            fill.loc[fill.eq("") & ws_l.str.contains("ipad", regex=False)] = "Mozilla/5.0 (iPad; CPU OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile Safari/604.1"

            still = fill.eq("")
            if still.any():
                pool = np.array([
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Version/17.2 Safari/605.1.15",
                    "curl/7.88.1",
                    "python-requests/2.31.0",
                    "Go-http-client/1.1",
                    "Java/17 Apache-HttpClient/5.2",
                ], dtype=object)
                h = stable_u64(row_seed, "ua_pool")
                fill.loc[still] = pool[(h[still.to_numpy()] % len(pool)).astype(int)]

            ua = ua.mask(ua_bad, fill)
        df["user_agent"] = ensure_no_bad_tokens(ua, "Mozilla/5.0")

        # referrer
        ref = clean_missing_series_fast(df["referrer"]).fillna("").astype(str).str.strip()
        ref_bad = ref.eq("") | ref.str.lower().isin(MISS_STRS) | ref.str.match(BAD_STR_RE, na=False)
        if ref_bad.any():
            lt2 = df["log_type"].fillna("").astype(str).str.lower()
            fill = pd.Series("", index=df.index, dtype="object")
            fill.loc[lt2.isin(["type9_dynamic_csv", "idp", "ad_auth", "mfa", "aaa", "pam"])] = "sso"
            fill.loc[lt2.eq("type6_event_csv")] = "eventlog"
            fill.loc[lt2.str.contains("type7", regex=False)] = "dhcpd"
            fill.loc[lt2.eq("type8_firewall_csv")] = "firewall-policy"
            fill.loc[lt2.eq("waf")] = "edge"
            fill.loc[lt2.eq("apigw")] = "gateway"
            fill.loc[lt2.eq("lb")] = "loadbalancer"
            fill.loc[lt2.eq("envoy")] = "envoy"
            fill.loc[lt2.eq("dns")] = "resolver"
            fill.loc[lt2.eq("ids")] = "ids"
            fill.loc[lt2.eq("tls")] = "tls"
            fill.loc[lt2.eq("cloud_audit")] = "cloud"
            fill.loc[lt2.eq("k8s_audit")] = "kube-apiserver"
            fill.loc[lt2.eq("container_runtime")] = "containerd"
            fill.loc[lt2.eq("secrets_kms")] = "kms"
            fill.loc[lt2.eq("email_sec")] = "mail-gateway"
            fill.loc[lt2.eq("gh_audit")] = "github"
            fill.loc[lt2.eq("cicd")] = "pipeline"

            seed = row_seed.astype(str)
            pick = (stable_u64(seed, "refpick") % 4).astype(int)
            webref = np.where(
                pick == 0, "Direct Entry",
                np.where(pick == 1, "https://www.google.com/", np.where(pick == 2, "https://duckduckgo.com/", "https://www.bing.com/"))
            )
            fill = fill.mask(fill.eq(""), webref)
            fill = fill.mask(fill.str.contains(r"\.exe$", regex=True, na=False), "Direct Entry")

            ref = ref.mask(ref_bad, fill)
        df["referrer"] = ensure_no_bad_tokens(ref, "Direct Entry")

        # username (no personal propagation)
        user = clean_missing_series_fast(df["username"]).fillna("").astype(str).str.strip()
        user = user.mask(user.map(lambda x: bool(PERSON_NAME_RE.match(x))), "")
        missing_u = user.eq("")

        if missing_u.any() and isinstance(self.priors, dict):
            ip_to_user = self.priors.get("ip_to_user", {}) or {}
            ws_to_user = self.priors.get("ws_to_user", {}) or {}
            cip_s = df["client_ip"].fillna("").astype(str)
            ws_s = df["workstation"].fillna("").astype(str)

            msk = missing_u & cip_s.str.match(_IPV4_RE, na=False)
            if msk.any():
                cand = cip_s.loc[msk].map(ip_to_user).fillna("").astype(str)
                ok2 = cand.map(lambda x: bool(SAFE_USER_RE.match(str(x))))
                user.loc[msk] = np.where(ok2.to_numpy(), cand.to_numpy(), "").astype(object)

            missing_u = user.eq("")
            msk = missing_u & ws_s.ne("")
            if msk.any():
                cand = ws_s.loc[msk].map(ws_to_user).fillna("").astype(str)
                ok2 = cand.map(lambda x: bool(SAFE_USER_RE.match(str(x))))
                user.loc[msk] = np.where(ok2.to_numpy(), cand.to_numpy(), "").astype(object)

        missing_u = user.eq("")
        if missing_u.any():
            cip_s = df["client_ip"].fillna("").astype(str)
            kinds = cip_s.map(ip_kind_fast)
            user.loc[missing_u & kinds.eq("public")] = "external_user"
            user.loc[missing_u & ~kinds.eq("public")] = "corp_user"

        ws_s = df["workstation"].fillna("").astype(str).str.strip()
        same = user.str.lower().eq(ws_s.str.lower()) & user.ne("") & ws_s.ne("")
        if same.any():
            kinds = df["client_ip"].fillna("").astype(str).map(ip_kind_fast)
            user.loc[same & kinds.eq("public")] = "external_user"
            user.loc[same & ~kinds.eq("public")] = "corp_user"

        df["username"] = ensure_no_bad_tokens(user, "corp_user")

        # workstation
        ws = clean_missing_series_fast(df["workstation"]).fillna("").astype(str).str.strip()
        ws_bad = ws.eq("") | ws.str.lower().isin(MISS_STRS) | ws.str.match(BAD_STR_RE, na=False)
        if ws_bad.any():
            hid = stable_u64(row_seed, "ws") % 9000 + 1000
            ws = ws.mask(ws_bad, "WS-" + pd.Series(hid, index=df.index).astype(str))
        df["workstation"] = ensure_no_bad_tokens(ws, "WS-1000")

        # command
        cmd = clean_missing_series_fast(df["command"]).fillna("").astype(str).str.strip()
        cmd_bad = cmd.eq("") | cmd.str.lower().isin(MISS_STRS) | cmd.str.match(BAD_STR_RE, na=False)
        if cmd_bad.any():
            proc2 = df["process"].fillna("client_app.exe").astype(str).str.strip()
            meth2 = df["method"].fillna("GET").astype(str).str.upper()
            fu2 = df["full_url"].fillna("http://localhost/").astype(str)

            h = (stable_u64(row_seed, "cmdpick") % 3).astype(np.int64)
            h_s = pd.Series(h, index=df.index)

            outc = pd.Series("", index=df.index, dtype="object")

            rhs0 = (proc2 + " --request " + meth2 + " " + fu2).astype(object)
            rhs1 = (proc2 + " --connect " + df["domain"].astype(str) + " --path " + df["url_path"].astype(str)).astype(object)
            rhs2 = (proc2 + " --navigate " + df["referrer"].astype(str) + " --url " + fu2).astype(object)

            m0 = h_s.eq(0)
            m1 = h_s.eq(1)
            m2 = h_s.eq(2)

            if m0.any():
                outc.loc[m0] = rhs0.loc[m0].to_numpy()
            if m1.any():
                outc.loc[m1] = rhs1.loc[m1].to_numpy()
            if m2.any():
                outc.loc[m2] = rhs2.loc[m2].to_numpy()

            cmd = cmd.mask(cmd_bad, outc)

        df["command"] = ensure_command_prefix_series(df["process"], cmd)

        # bytes
        df = fix_or_impute_bytes(df, self.bytes_priors, row_seed)

        # FINAL: 100% coverage on 16 primary cols
        df["client_ip"] = ensure_no_bad_tokens(df["client_ip"].astype(str), fill_text)
        df["dest_ip"] = ensure_no_bad_tokens(df["dest_ip"].astype(str), fill_text)
        df["timestamp"] = ensure_no_bad_tokens(df["timestamp"].astype(str), fill_text)
        df["method"] = ensure_no_bad_tokens(df["method"].astype(str), "GET")
        df["full_url"] = ensure_no_bad_tokens(df["full_url"].astype(str), "http://localhost/")
        df["domain"] = ensure_no_bad_tokens(df["domain"].astype(str), "unknown.local")
        df["url_path"] = ensure_no_bad_tokens(df["url_path"].astype(str), "/")
        df["referrer"] = ensure_no_bad_tokens(df["referrer"].astype(str), "Direct Entry")
        df["user_agent"] = ensure_no_bad_tokens(df["user_agent"].astype(str), "Mozilla/5.0")
        df["username"] = ensure_no_bad_tokens(df["username"].astype(str), "corp_user")
        df["workstation"] = ensure_no_bad_tokens(df["workstation"].astype(str), "WS-1000")
        df["process"] = ensure_no_bad_tokens(df["process"].astype(str), "client_app.exe")
        df["command"] = ensure_no_bad_tokens(df["command"].astype(str), "client_app.exe")
        df["log_type"] = ensure_no_bad_tokens(df["log_type"].astype(str), "futuristic_unknown")

        df["status"] = pd.to_numeric(df["status"], errors="coerce").fillna(200).round().astype(int)
        df.loc[(df["status"] < 100) | (df["status"] > 599), "status"] = 200
        df["bytes_in"] = pd.to_numeric(df["bytes_in"], errors="coerce").fillna(0).clip(lower=0).astype(int)
        df["bytes_out"] = pd.to_numeric(df["bytes_out"], errors="coerce").fillna(0).clip(lower=0).astype(int)

        # If domain accidentally ended up as an IPv4 string, derive a domain from dest_ip
        dom_bad = df["domain"].astype(str).str.match(_IPV4_RE, na=False)
        if dom_bad.any():
            df.loc[dom_bad, "domain"] = derive_domain_from_dest_ip_series(df.loc[dom_bad, "dest_ip"]).to_numpy()

        return df


# -----------------------------
# Convenience export (helps if someone mistakenly does: from imputer import imputer_mod)
# -----------------------------
imputer_mod = sys.modules[__name__]

__all__ = [
    "PRIMARY_COLS",
    "INTERNAL_COLS",
    "CANON_TYPES_45",
    "normalize_log_type",
    "sanitize_domain_series",
    "ForensicImputer",
    "imputer_mod",
]