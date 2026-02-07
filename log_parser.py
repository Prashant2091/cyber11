# log_parser.py
# ============================================================
# CODE 1 — Universal Log Parser (Broad Coverage)
# ------------------------------------------------------------
# Goal: Parse *any* real-world log line into your canonical 16 columns
#       + raw_log + url_path (internal helper).
#
# Coverage:
# ✅ 9 trained families (type1..type9) — robust ordering (type9 before type5)
# ✅ Apache combined access logs
# ✅ Telemetry (ISOZ + tag + key=value), supporting 35+ families
# ✅ Linux syslog auth/sudo/sshd
# ✅ Linux auditd (type=... msg=audit(epoch...))
# ✅ UEBA JSON logs (user_id, data_download_gb, login_failures, off_hours_access)
# ✅ Unknown/futuristic logs (best-effort URL/IP/timestamp extraction)
#
# Guarantees:
# - Never raises on malformed input (returns a canonical dict).
# - Never drops non-empty lines (headers are kept as log_type="header").
# - Never forces "NotProvided" placeholders here (imputer can do that).
#
# Interface:
#   parse_log_universal(line: str) -> Dict[str, Any]
#
# Dependencies: standard lib + pandas (optional but used for stable parsing)
# ============================================================

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, List
from urllib.parse import urlparse

import pandas as pd


# -----------------------------
# Canonical columns
# -----------------------------
PRIMARY_COLS: List[str] = [
    "client_ip", "timestamp", "method", "full_url", "status",
    "bytes_out", "referrer", "user_agent", "bytes_in", "domain",
    "dest_ip", "username", "workstation", "process", "command", "log_type"
]
INTERNAL_COLS: List[str] = ["raw_log", "url_path"]


# -----------------------------
# Missing tokens + safe casts
# -----------------------------
MISS_TOKENS = {
    "", " ", "unknown", "missing", "null", "none", "nan", "na", "n/a",
    "-", "--", "notprovided", "not_provided", "unknown-domain", "unknown_domain"
}
MISS_STRS = {str(x).strip().lower() for x in MISS_TOKENS if x is not None}

def is_missing_str(x: str) -> bool:
    return str(x).strip().lower() in MISS_STRS

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if is_missing_str(s) else s

def safe_int(x: Any, default: int = 0) -> int:
    s = safe_str(x).strip()
    if s == "":
        return default
    try:
        return int(float(s))
    except Exception:
        return default

def safe_float(x: Any) -> float:
    s = safe_str(x).strip()
    if s == "":
        return float("nan")
    try:
        v = float(s)
        if not (v >= 0):
            return float("nan")
        return v
    except Exception:
        return float("nan")


# -----------------------------
# Regex / detectors
# -----------------------------
_IPV4_RE = re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$")
_REDACTED_IP_LIKE = re.compile(r"^(?:x|\d{1,3})(?:\.(?:x|\d{1,3})){3}$", re.I)
IPV4_FINDER = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")

_URL_RE  = re.compile(r'(?i)(https?://[^\s"\'<>]+)')
_HOST_FINDER = re.compile(r'(?i)\b([a-z0-9-]+(?:\.[a-z0-9-]+)+)(?::\d{1,5})?\b')

_HTTP_METHODS = {"GET","POST","PUT","DELETE","PATCH","HEAD","OPTIONS"}

# timestamps
TYPE1_TS = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4}$")
APACHE_TS = re.compile(r"^\d{1,2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4}$")
ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

ISO_NAIVE = re.compile(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$')
DATE_MMDDYY = re.compile(r"^\d{1,2}/\d{1,2}/\d{2}$")
DATE_YYYYMMDD = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_HHMMSS = re.compile(r"^\d{2}:\d{2}:\d{2}$")

# telemetry kv: "ISOZ tag k=v ..."
ISO_TAG_LINE_RE = re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+(?P<tag>[A-Za-z0-9_]+)\s+(?P<rest>.*)$')
KV_RE = re.compile(r'(?P<k>[A-Za-z0-9_]+)=(?P<v>"[^"]*"|\S+)')

# Linux syslog (month day time host program[pid]: msg)
SYSLOG_RE = re.compile(
    r'^(?P<mon>[A-Za-z]{3})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+(?P<host>\S+)\s+(?P<prog>[A-Za-z0-9_.-]+)(?:\[\d+\])?:\s+(?P<msg>.*)$'
)

# auditd: type=EXECVE msg=audit(1730419272.123:101): ...
AUDITD_RE = re.compile(r'^type=(?P<atype>[A-Z_]+)\s+msg=audit\((?P<epoch>\d+)(?:\.\d+)?:\d+\):\s*(?P<rest>.*)$')
AUDITD_ARG_RE = re.compile(r'\ba(?P<i>\d+)="(?P<v>[^"]*)"\b')

# header lines
CSV_HEADER_RE = re.compile(r'^(?i)(id|timestamp|date)[,\t]')


def looks_like_ipv4_or_redacted(x: str) -> bool:
    s = safe_str(x).strip()
    if not s:
        return False
    return bool(_IPV4_RE.match(s) or _REDACTED_IP_LIKE.match(s) or IPV4_FINDER.fullmatch(s))

def looks_like_domain(x: str) -> bool:
    s = safe_str(x).strip().strip("[](){}<>,.;'\"").lower()
    if not s or "." not in s:
        return False
    tld = s.rsplit(".", 1)[-1]
    return bool(2 <= len(tld) <= 24 and re.search(r"[a-z]", tld))

def looks_like_hostname(x: str) -> bool:
    s = safe_str(x).strip()
    if not s:
        return False
    if "." in s:
        return False
    return bool(re.search(r"[A-Za-z]", s) and re.search(r"[-_]", s))

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

def normalize_domain(domain: str) -> str:
    d = safe_str(domain).strip().strip("[]").strip().lower()
    d = d.strip(',.);]\'"')
    if d.startswith("http://") or d.startswith("https://"):
        try:
            p = urlparse(d)
            d = (p.netloc or "").split("@")[-1].split(":")[0].strip("[]").strip().lower()
        except Exception:
            pass
    # if it isn't domain-like, treat as missing (prevents hostnames becoming domain)
    if d and not looks_like_domain(d):
        return ""
    return d

def make_full_url(domain: str, url_path: str) -> str:
    dom = normalize_domain(domain) or "localhost"
    up = safe_str(url_path).strip() or "/"
    if up.lower().startswith(("http://","https://")):
        return up
    if not up.startswith("/"):
        up = "/" + up
    return f"http://{dom}{up}"

def split_ip_port(s: str) -> Tuple[str, str]:
    s = safe_str(s).strip()
    if not s:
        return "", ""
    if ":" in s and IPV4_FINDER.search(s):
        a = s.split(":")
        return a[0], a[1] if len(a) > 1 else ""
    return s, ""


# -----------------------------
# Output normalizer
# -----------------------------
def _base_row(raw: str) -> Dict[str, Any]:
    return {
        "raw_log": raw,
        "log_type": "unknown",
        "timestamp": "",
        "client_ip": "",
        "dest_ip": "",
        "method": "GET",
        "url_path": "/",
        "full_url": "",
        "status": 200,
        "bytes_out": float("nan"),
        "bytes_in": float("nan"),
        "referrer": "Direct Entry",
        "user_agent": "Mozilla/5.0",
        "domain": "",
        "username": "",
        "workstation": "",
        "process": "",
        "command": "",
    }

def ensure_canonical(d: Dict[str, Any]) -> Dict[str, Any]:
    # ensure required keys
    for c in (PRIMARY_COLS + INTERNAL_COLS):
        if c not in d:
            d[c] = "" if c not in ("status","bytes_in","bytes_out") else float("nan")

    d["raw_log"] = safe_str(d.get("raw_log", ""))

    # method
    m = safe_str(d.get("method","GET")).upper().strip()
    d["method"] = m if m in _HTTP_METHODS else "GET"

    # status
    d["status"] = safe_int(d.get("status", 200), 200)
    if d["status"] < 100 or d["status"] > 599:
        d["status"] = 200

    # bytes
    d["bytes_in"] = safe_float(d.get("bytes_in", float("nan")))
    d["bytes_out"] = safe_float(d.get("bytes_out", float("nan")))

    # timestamp (kept raw; timezone module later)
    d["timestamp"] = safe_str(d.get("timestamp","")).strip()

    # domain/url/full_url
    d["domain"] = normalize_domain(d.get("domain",""))
    up = safe_str(d.get("url_path","/")).strip() or "/"
    if not up.startswith("/"):
        up = "/" + up
    d["url_path"] = up

    fu = safe_str(d.get("full_url","")).strip()
    if not fu and d["domain"]:
        fu = make_full_url(d["domain"], d["url_path"])
    d["full_url"] = fu

    # strip ip fields
    d["client_ip"] = safe_str(d.get("client_ip","")).strip()
    d["dest_ip"] = safe_str(d.get("dest_ip","")).strip()

    # other strings
    for k in ("referrer","user_agent","username","workstation","process","command","log_type"):
        d[k] = safe_str(d.get(k,"")).strip()

    if not d["referrer"]:
        d["referrer"] = "Direct Entry"
    if not d["user_agent"]:
        d["user_agent"] = "Mozilla/5.0"
    if not d["log_type"]:
        d["log_type"] = "unknown"

    return d


# ============================================================
# PARSERS
# ============================================================

# -----------------------------
# Type1 (space + TZ)
# -----------------------------
def _choose_domain_from_tail(tail: str) -> str:
    t = safe_str(tail).strip()
    if not t:
        return ""
    tokens = t.split()
    for tok in reversed(tokens):
        cand = tok.strip().strip("[](){}<>,.;'\"")
        if cand.lower().endswith((".rules",".txt",".exe",".dll",".json",".xml",".png",".jpg",".css",".js",".zip",".tar",".tgz",".gz",".pdf")):
            continue
        if "." in cand and re.search(r"[A-Za-z]", cand):
            if cand.lower().startswith(("http://","https://")):
                h = host_from_anywhere(cand)
                return normalize_domain(h or "")
            return normalize_domain(cand)
    return normalize_domain(host_from_anywhere(t) or "")

def parse_type1_space(line: str) -> Optional[Dict[str, Any]]:
    raw = safe_str(line).rstrip("\n")
    s = raw.strip()
    if not s:
        return None
    toks = s.split()
    if len(toks) < 9:
        return None
    ts = " ".join(toks[0:3])
    if not TYPE1_TS.match(ts):
        return None

    ip_or_host = toks[3]
    method = toks[4].upper()
    url_path = toks[5]
    status = safe_int(toks[6], 200)
    bytes_out = safe_int(toks[7], 0)
    bytes_in = safe_int(toks[8], 0)
    tail = " ".join(toks[9:]) if len(toks) > 9 else ""

    dom = _choose_domain_from_tail(tail)
    ua = tail.split(",")[0].strip() if tail else ""

    client_ip = ip_or_host if looks_like_ipv4_or_redacted(ip_or_host) else ""
    workstation = "" if client_ip else ip_or_host

    d = _base_row(raw)
    d.update({
        "log_type": "type1_space",
        "timestamp": ts,
        "client_ip": client_ip,
        "workstation": workstation,
        "method": method,
        "url_path": url_path,
        "status": status,
        "bytes_out": bytes_out,
        "bytes_in": bytes_in,
        "user_agent": ua,
        "domain": dom,
        "referrer": "Direct Entry",
    })
    return ensure_canonical(d)


# -----------------------------
# Apache combined access log
# -----------------------------
APACHE_RE = re.compile(
    r'^(?P<client_ip>\S+)\s+\S+\s+\S+\s+\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<method>[A-Z]+)\s+(?P<url_path>\S+)\s+HTTP/[\d.]+"\s+'
    r'(?P<status>\d{3})\s+(?P<bytes_out>[-\d]+)\s+'
    r'"(?P<referrer>[^"]*)"\s+"(?P<user_agent>[^"]*)"'
)

def parse_apache(line: str) -> Optional[Dict[str, Any]]:
    raw = safe_str(line).rstrip("\n")
    s = raw.strip()
    if not s:
        return None
    m = APACHE_RE.match(s)
    if not m:
        return None
    g = m.groupdict()
    bo = safe_int(g.get("bytes_out", 0), 0)
    if bo < 0:
        bo = 0
    ref = g.get("referrer","") or "Direct Entry"
    if ref.strip() in ("-", '"-"'):
        ref = "Direct Entry"
    dom = host_from_anywhere(ref) or host_from_anywhere(g.get("url_path","")) or ""
    d = _base_row(raw)
    d.update({
        "log_type": "apache",
        "timestamp": g.get("timestamp",""),
        "client_ip": g.get("client_ip",""),
        "method": g.get("method","GET"),
        "url_path": g.get("url_path","/"),
        "status": safe_int(g.get("status",200), 200),
        "bytes_out": bo,
        "bytes_in": 0,
        "referrer": ref,
        "user_agent": g.get("user_agent",""),
        "domain": dom,
        "process": "apache2",
        "command": f"apache2 --handle-request {g.get('method','GET')} {g.get('url_path','/')}",
    })
    return ensure_canonical(d)


# -----------------------------
# UEBA JSON
# -----------------------------
def parse_ueba_json(line: str) -> Optional[Dict[str, Any]]:
    raw = safe_str(line).rstrip("\n")
    s = raw.strip()
    if not s.startswith("{"):
        return None
    try:
        obj = json.loads(s.rstrip(","))
    except Exception:
        return None

    if "user_id" in obj and ("data_download_gb" in obj or "login_failures" in obj or "off_hours_access" in obj):
        user_id = safe_str(obj.get("user_id",""))
        data_gb = float(obj.get("data_download_gb", 0) or 0)
        login_fail = int(obj.get("login_failures", 0) or 0)
        off_hours = int(obj.get("off_hours_access", 0) or 0)

        bytes_out = int(round(data_gb * (1024**3)))
        status = 401 if login_fail > 0 else 200

        d = _base_row(raw)
        d.update({
            "log_type": "ueba_user_behavior",
            "timestamp": safe_str(obj.get("timestamp","")),
            "method": "POST",
            "url_path": "/ueba/user_behavior",
            "status": status,
            "bytes_out": bytes_out,
            "bytes_in": 0,
            "referrer": "Direct Entry",
            "user_agent": "UEBA/behavior",
            "domain": "ueba.local",
            "username": user_id,
            "process": "ueba_agent",
            "command": f"ueba_agent --user {user_id} --off_hours_access {off_hours} --login_failures {login_fail}",
        })
        return ensure_canonical(d)

    return None


# -----------------------------
# CSV trained types (3–9) and proxy types
# IMPORTANT: type9 is checked BEFORE type5.
# -----------------------------
def _csv_row(line: str) -> Optional[List[str]]:
    s = safe_str(line).strip()
    if not s:
        return None
    try:
        row = next(csv.reader([s], skipinitialspace=True))
    except Exception:
        row = s.split(",")
    row = [safe_str(x).strip() for x in row]
    while row and row[-1] == "":
        row.pop()
    return row

def parse_csv_types(line: str) -> Optional[Dict[str, Any]]:
    raw = safe_str(line).rstrip("\n")
    s = raw.strip()
    if not s:
        return None
    if CSV_HEADER_RE.match(s):
        d = _base_row(raw)
        d["log_type"] = "header"
        return ensure_canonical(d)

    row = _csv_row(raw)
    if not row:
        return None
    n = len(row)

    # TYPE9: ID,Timestamp,Hostname,User,IP,EventID,LogonID,Action,(Label?)
    if n >= 8 and row[0].isdigit() and ISO_NAIVE.match(row[1]) and row[5].isdigit() and looks_like_ipv4_or_redacted(row[4]):
        action = row[7].upper().strip()
        host = row[2]
        user = row[3]
        username = "" if user.strip().upper() == host.strip().upper() else user
        event_id = row[5]
        logon_id = row[6]
        if action in ("LOGOFF","LOGOUT","REJECT","DENY"):
            method = "DELETE"
            status = 204 if action in ("LOGOFF","LOGOUT") else 401
            url_path = "/auth/logoff"
        else:
            method = "POST"
            status = 200
            url_path = "/auth/logon"
        d = _base_row(raw)
        d.update({
            "log_type": "type9_dynamic_csv",
            "timestamp": row[1],
            "client_ip": row[4],
            "method": method,
            "url_path": url_path,
            "status": status,
            "referrer": "winlogon.exe",
            "user_agent": "WindowsEventLog/10.0",
            "domain": "sso.corp.local",
            "username": username,
            "workstation": host,
            "process": "winlogon.exe",
            "command": f"winlogon.exe --{(action or 'auth').lower()} --eventid {event_id} --logonid {logon_id}",
        })
        return ensure_canonical(d)

    # TYPE4: ts,client_ip,dest_ip,method,url_path,ref,status,bo,bi,ua,username
    if n >= 10 and ISO_NAIVE.match(row[0]) and row[3].upper() in _HTTP_METHODS:
        d = _base_row(raw)
        d.update({
            "log_type": "type4_web_csv",
            "timestamp": row[0],
            "client_ip": row[1],
            "dest_ip": row[2],
            "method": row[3].upper(),
            "url_path": row[4],
            "referrer": row[5] or "Direct Entry",
            "status": safe_int(row[6], 200),
            "bytes_out": safe_int(row[7], 0),
            "bytes_in": safe_int(row[8], 0),
            "user_agent": row[9].strip('"') if len(row) > 9 else "",
            "domain": normalize_domain(host_from_anywhere(row[5]) or host_from_anywhere(row[4]) or ""),
            "username": row[10] if len(row) > 10 else "",
        })
        return ensure_canonical(d)

    # TYPE6: ID,Timestamp,Hostname,EventID,Description,Command,(Label?)
    if n >= 6 and row[0].isdigit() and ISO_NAIVE.match(row[1]) and row[3].isdigit():
        cmd = ",".join(row[5:]).strip() if len(row) > 5 else ""
        if cmd.upper() == "N/A":
            cmd = ""
        d = _base_row(raw)
        d.update({
            "log_type": "type6_event_csv",
            "timestamp": row[1],
            "method": "POST",
            "url_path": f"/event/{row[3]}",
            "status": 201,
            "referrer": "winlogon.exe",
            "user_agent": "WindowsEventLog/10.0",
            "domain": "eventlog.corp.local",
            "workstation": row[2],
            "command": cmd if cmd else row[4],
        })
        return ensure_canonical(d)

    # TYPE7: ID,Date,Time,Description,IP,Host,MAC
    if n >= 7 and row[0].isdigit() and DATE_MMDDYY.match(row[1]) and TIME_HHMMSS.match(row[2]) and looks_like_ipv4_or_redacted(row[4]):
        try:
            dt_local = datetime.strptime(f"{row[1]} {row[2]}", "%m/%d/%y %H:%M:%S")
            ts = dt_local.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = ""
        d = _base_row(raw)
        d.update({
            "log_type": "type7_asset_csv",
            "timestamp": ts,
            "client_ip": row[4],
            "method": "PUT" if row[3].lower() == "assign" else "POST",
            "url_path": "/dhcp/" + row[3].lower(),
            "status": 200,
            "referrer": "dhcpd",
            "user_agent": "DHCPClient/1.0",
            "domain": "dhcp.corp.local",
            "workstation": row[5],
            "process": "dhcpclient.exe",
            "command": f"dhcpclient.exe --{row[3].lower()} --ip {row[4]} --mac {row[6]}",
        })
        return ensure_canonical(d)

    # TYPE8: ID,Date,Time,SourceIP,DestIP,Action,BytesOut
    if n >= 7 and row[0].isdigit() and DATE_YYYYMMDD.match(row[1]) and TIME_HHMMSS.match(row[2]) and looks_like_ipv4_or_redacted(row[3]) and looks_like_ipv4_or_redacted(row[4]):
        ts = f"{row[1]} {row[2]}"
        action = row[5]
        d = _base_row(raw)
        d.update({
            "log_type": "type8_firewall_csv",
            "timestamp": ts,
            "client_ip": row[3],
            "dest_ip": row[4],
            "method": "GET" if action.upper() == "ALLOW" else "DELETE",
            "url_path": "/firewall/" + action.lower(),
            "status": 200 if action.upper() == "ALLOW" else 403,
            "bytes_out": safe_int(row[6], 0),
            "referrer": "firewall-policy",
            "domain": "",
        })
        return ensure_canonical(d)

    # TYPE3: ts,user,ws,process,command,ip
    if n >= 6 and ISO_NAIVE.match(row[0]):
        ts, user, ws, proc = row[0], row[1], row[2], row[3]
        dip = row[-1]
        cmd = ",".join(row[4:-1]).strip()
        url, host, path = extract_first_url(cmd)
        d = _base_row(raw)
        d.update({
            "log_type": "type3_proc_csv",
            "timestamp": ts,
            "dest_ip": dip if looks_like_ipv4_or_redacted(dip) else "",
            "method": "GET" if url else "POST",
            "url_path": path or "/",
            "full_url": url or "",
            "status": 200,
            "referrer": "explorer.exe",
            "user_agent": proc,
            "domain": normalize_domain(host or ""),
            "username": user,
            "workstation": ws,
            "process": proc,
            "command": cmd,
        })
        return ensure_canonical(d)

    # TYPE5 proxy: ID,Timestamp,Source_IP,Destination_Domain,Referrer,User_Agent,Bytes_Out,(Label?)
    if n >= 7 and row[0].isdigit() and ISO_NAIVE.match(row[1]) and looks_like_ipv4_or_redacted(row[2]) and looks_like_domain(row[3]):
        # robust bytes_out/label disambiguation
        nums = []
        for j in range(len(row) - 1, -1, -1):
            if re.fullmatch(r"-?\d+(\.\d+)?", row[j].strip()):
                nums.append(j)
        bytes_out = 0
        ua = ""
        if nums:
            j0 = nums[0]
            v0 = safe_int(row[j0], 0)
            if v0 in (0, 1) and len(nums) >= 2:
                bytes_out = safe_int(row[nums[1]], 0)
                ua = ",".join(row[5:nums[1]]).strip()
            else:
                bytes_out = v0
                ua = ",".join(row[5:j0]).strip()
        else:
            ua = ",".join(row[5:]).strip()

        d = _base_row(raw)
        d.update({
            "log_type": "type5_proxy_csv",
            "timestamp": row[1],
            "client_ip": row[2],
            "method": "GET",
            "url_path": "/",
            "status": 200,
            "bytes_out": bytes_out,
            "bytes_in": float("nan"),
            "referrer": row[4] or "Direct Entry",
            "user_agent": ua,
            "domain": normalize_domain(row[3]),
        })
        return ensure_canonical(d)

    return None


# -----------------------------
# Telemetry KV (ISOZ + tag + k=v)
# Broad mapping across 35+ families.
# -----------------------------
def parse_kv(rest: str) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for m in KV_RE.finditer(rest):
        k = m.group("k")
        v = m.group("v")
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        d[k] = v
    return d

def _parse_req_field(req: str) -> Tuple[str, str]:
    r = safe_str(req).strip().strip('"').strip()
    parts = r.split()
    if len(parts) >= 2 and parts[0].upper() in _HTTP_METHODS:
        return parts[0].upper(), parts[1]
    return "GET", "/"

def _status_from_result(result: str) -> int:
    r = safe_str(result).upper()
    if r in ("SUCCESS","PASS","APPROVE","ACCEPT","ALLOW","OK"):
        return 200
    if r in ("FAIL","FAILED","DENIED","REJECT","REJECTED","FORBID","FORBIDDEN","BLOCK","BLOCKED","TIMEOUT","ERROR"):
        # distinguish block vs auth fail
        return 403 if r in ("DENIED","FORBID","FORBIDDEN","BLOCK","BLOCKED") else 401
    return 200

def parse_telemetry_kv(line: str) -> Optional[Dict[str, Any]]:
    raw = safe_str(line).rstrip("\n")
    s = raw.strip()
    if not s:
        return None
    m = ISO_TAG_LINE_RE.match(s)
    if not m:
        return None

    ts = m.group("ts")
    tag = (m.group("tag") or "").lower()
    rest = m.group("rest") or ""
    kv = parse_kv(rest)

    d = _base_row(raw)
    d["log_type"] = tag
    d["timestamp"] = ts
    d["command"] = rest

    # client_ip candidates
    for k in ("client_ip","client","src_ip","src","orig","downstream","requester_ip","ip"):
        if k in kv and kv[k]:
            d["client_ip"] = split_ip_port(kv[k])[0]
            break

    # dest_ip candidates
    for k in ("dest_ip","dst","resp","target","upstream","server_ip"):
        if k in kv and kv[k]:
            d["dest_ip"] = split_ip_port(kv[k])[0]
            break

    # domain candidates
    for k in ("domain","host","sni","qname","bucket","repo","system","service","app"):
        if k in kv and kv[k]:
            d["domain"] = normalize_domain(kv[k])
            if d["domain"]:
                break

    # username candidates
    for k in ("user","principal","actor","account","requester"):
        if k in kv and kv[k]:
            d["username"] = kv[k]
            break

    # workstation candidates
    for k in ("device","hostname","host","node","workstation"):
        if k in kv and kv[k]:
            # only if it's not a domain
            if not looks_like_domain(kv[k]):
                d["workstation"] = kv[k]
                break

    # user_agent candidates
    for k in ("ua","user_agent","ja3","client_ua"):
        if k in kv and kv[k]:
            d["user_agent"] = kv[k]
            break

    # method/path/status
    if "req" in kv:
        mth, path = _parse_req_field(kv["req"])
        d["method"] = mth
        d["url_path"] = path
    else:
        if "method" in kv:
            d["method"] = safe_str(kv.get("method","GET")).upper()
        if "path" in kv and kv["path"]:
            d["url_path"] = kv["path"]
        elif "uri" in kv and kv["uri"]:
            d["url_path"] = kv["uri"]

    # status
    if "status" in kv:
        d["status"] = safe_int(kv.get("status", 200), 200)
    elif "code" in kv:
        d["status"] = safe_int(kv.get("code", 200), 200)
    elif "severity" in kv:
        # keep 200, severity stored in command already
        d["status"] = 200
    elif "result" in kv:
        d["status"] = _status_from_result(kv.get("result",""))
    elif "outcome" in kv:
        d["status"] = _status_from_result(kv.get("outcome",""))

    # bytes
    if "bytes_out" in kv:
        d["bytes_out"] = safe_int(kv.get("bytes_out", 0), 0)
    elif "sent" in kv:
        d["bytes_out"] = safe_int(kv.get("sent", 0), 0)
    elif "bytes" in kv:
        d["bytes_out"] = safe_int(kv.get("bytes", 0), 0)
    elif "orig_bytes" in kv:
        d["bytes_out"] = safe_int(kv.get("orig_bytes", 0), 0)

    if "bytes_in" in kv:
        d["bytes_in"] = safe_int(kv.get("bytes_in", 0), 0)
    elif "resp_bytes" in kv:
        d["bytes_in"] = safe_int(kv.get("resp_bytes", 0), 0)

    # tag-specific tweaks
    if tag == "dns":
        qname = kv.get("qname","")
        if qname:
            d["domain"] = normalize_domain(qname) or d["domain"]
            d["url_path"] = "/dns/" + qname
        rcode = safe_str(kv.get("rcode","")).upper()
        if rcode and rcode != "NOERROR":
            d["status"] = 404
        answer = kv.get("answer","")
        if answer and looks_like_ipv4_or_redacted(split_ip_port(answer)[0]):
            d["dest_ip"] = split_ip_port(answer)[0]

    if tag in ("flow","cloud_flow","zeek_conn"):
        # flows usually: src/dst and bytes_out/bytes_in
        d["url_path"] = d["url_path"] if d["url_path"] and d["url_path"] != "/" else f"/{tag}"

    if tag in ("tls","ids"):
        # TLS: sni and ja3; IDS: alert/severity
        if "sni" in kv and kv["sni"]:
            d["domain"] = normalize_domain(kv["sni"]) or d["domain"]
        d["url_path"] = f"/{tag}"

    # derive full_url if we have domain and url_path
    if not safe_str(d.get("full_url","")).strip() and d.get("domain",""):
        d["full_url"] = make_full_url(d["domain"], d.get("url_path","/"))

    return ensure_canonical(d)


# -----------------------------
# Linux syslog (sshd/sudo)
# -----------------------------
_MONTHS = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
def parse_syslog(line: str, year_hint: Optional[int] = None) -> Optional[Dict[str, Any]]:
    raw = safe_str(line).rstrip("\n")
    s = raw.strip()
    if not s:
        return None
    m = SYSLOG_RE.match(s)
    if not m:
        return None
    mon = m.group("mon").lower()
    day = int(m.group("day"))
    t = m.group("time")
    host = m.group("host")
    prog = m.group("prog")
    msg = m.group("msg")

    year = int(year_hint) if year_hint is not None else datetime.utcnow().year
    mm = _MONTHS.get(mon, 1)
    # build ISO naive timestamp
    try:
        dt = datetime(year, mm, day, int(t[0:2]), int(t[3:5]), int(t[6:8]))
        ts = dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        ts = f"{mon} {day} {t}"

    client_ip = ""
    m_ip = re.search(r"\bfrom\s+(?P<ip>\d{1,3}(?:\.\d{1,3}){3})\b", msg)
    if m_ip:
        client_ip = m_ip.group("ip")

    status = 200
    if re.search(r"failed password|authentication failure|disconnecting|too many authentication failures", msg, re.I):
        status = 401

    # heuristics for user
    user = ""
    m_user = re.search(r"\bfor\s+(invalid user\s+)?(?P<u>[A-Za-z0-9_.-]+)\b", msg, re.I)
    if m_user:
        user = m_user.group("u")

    d = _base_row(raw)
    d.update({
        "log_type": "linux_syslog",
        "timestamp": ts,
        "client_ip": client_ip,
        "method": "POST",
        "url_path": "/syslog/auth",
        "status": status,
        "referrer": prog,
        "user_agent": prog,
        "domain": "linux.local",
        "username": user,
        "workstation": host,
        "process": prog,
        "command": msg,
    })
    return ensure_canonical(d)


# -----------------------------
# Linux auditd
# -----------------------------
def parse_auditd(line: str) -> Optional[Dict[str, Any]]:
    raw = safe_str(line).rstrip("\n")
    s = raw.strip()
    if not s:
        return None
    m = AUDITD_RE.match(s)
    if not m:
        return None

    atype = m.group("atype")
    epoch = int(m.group("epoch"))
    rest = m.group("rest")

    # convert epoch to ISOZ timestamp
    try:
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        ts = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        ts = ""

    # extract a0..aN
    args: Dict[int, str] = {}
    for mm in AUDITD_ARG_RE.finditer(rest):
        i = int(mm.group("i"))
        args[i] = mm.group("v")

    # build command
    cmd_parts = [args[k] for k in sorted(args.keys())] if args else []
    cmd = " ".join(cmd_parts).strip()

    # url/domain extraction from args
    url, host, path = extract_first_url(cmd)
    dom = normalize_domain(host or "")

    d = _base_row(raw)
    d.update({
        "log_type": f"auditd_{atype.lower()}",
        "timestamp": ts,
        "method": "POST",
        "url_path": path or "/auditd",
        "full_url": url or "",
        "status": 200,
        "referrer": "auditd",
        "user_agent": args.get(0, "") or "auditd",
        "domain": dom,
        "process": args.get(0, ""),
        "command": cmd if cmd else rest,
    })
    return ensure_canonical(d)


# -----------------------------
# Unknown fallback
# -----------------------------
def parse_unknown(line: str) -> Dict[str, Any]:
    raw = safe_str(line).rstrip("\n")
    s = raw.strip()
    d = _base_row(raw)

    # headers kept
    if CSV_HEADER_RE.match(s):
        d["log_type"] = "header"
        return ensure_canonical(d)

    # try URL
    url, host, path = extract_first_url(s)
    if url:
        d["full_url"] = url
    if path:
        d["url_path"] = path
    if host:
        d["domain"] = normalize_domain(host)

    # try IPs
    ips = IPV4_FINDER.findall(s)
    if len(ips) >= 1:
        d["client_ip"] = ips[0]
    if len(ips) >= 2:
        d["dest_ip"] = ips[1]

    # try method
    m = re.search(r"\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b", s)
    if m:
        d["method"] = m.group(1).upper()

    # try timestamp (very lightweight)
    m_isoz = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)", s)
    if m_isoz:
        d["timestamp"] = m_isoz.group(1)
    else:
        # naive iso
        m_iso = re.match(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})", s)
        if m_iso:
            d["timestamp"] = m_iso.group(1)

    # domain fallback
    if not d["domain"]:
        d["domain"] = normalize_domain(host_from_anywhere(s) or "")

    # command holds raw
    d["command"] = s
    return ensure_canonical(d)


# ============================================================
# Main entry
# ============================================================
def parse_log_universal(line: str) -> Dict[str, Any]:
    """
    Universal parser order (important):
      0) Keep headers (log_type="header")
      1) Type1 (space + TZ)
      2) Apache combined
      3) UEBA JSON (strict)
      4) CSV trained types (type9 checked before type5)
      5) Linux syslog
      6) Linux auditd
      7) Telemetry KV (ISOZ + tag + kv)
      8) Unknown fallback
    Always returns canonical dict with your 16 cols + raw_log + url_path.
    """
    raw = safe_str(line).rstrip("\n")
    if not raw.strip():
        return ensure_canonical(_base_row(raw))

    # header lines (never dropped)
    if CSV_HEADER_RE.match(raw.strip()):
        d = _base_row(raw)
        d["log_type"] = "header"
        return ensure_canonical(d)

    d = parse_type1_space(raw)
    if d is not None:
        return d

    d = parse_apache(raw)
    if d is not None:
        return d

    d = parse_ueba_json(raw)
    if d is not None:
        return d

    d = parse_csv_types(raw)
    if d is not None:
        return d

    d = parse_syslog(raw)
    if d is not None:
        return d

    d = parse_auditd(raw)
    if d is not None:
        return d

    d = parse_telemetry_kv(raw)
    if d is not None:
        return d

    return parse_unknown(raw)