# utils/app.py
# ============================================================
# CyberSecurity Log Classifier — FINAL HARDENED APP (NO-LOSS)
# ------------------------------------------------------------
# ✅ set_page_config called once and first Streamlit command
# ✅ robust local artifacts.py import (prevents wrong "artifacts" package)
# ✅ bundle fallback loader if artifacts.load_bundle fails/missing
# ✅ runtime patch for imputer.sanitize_domain_series (kills "~ float" crash even if imputer.py is old)
# ✅ odd-hours from RAW local wall-time (23:00–05:30) with UTC fallback ONLY for suspicious/bot/vpn contexts
# ✅ confidential primary conditions (regex/critical/ops + ipv6 tunnel combos) override classification always
# ✅ whitelist upload + modes: Off/Soft/Medium/Hard/Custom
#    - dampens probability ONLY when whitelist_hit AND NOT suspicious_context
#    - logs whitelist_hit/mode/factor/dampened in results + PDF
# ✅ alerts computed from odd-hours USED (local unless suspicious) + override-aware logic
# ✅ SHAP: never passes model=None; picks best available tree model; subset parsing supports ranges
# ✅ SHAP contributions (%): row uses |sv|; subset/all uses mean(|sv|) for magnitude + sign(mean(sv)) for direction
# ✅ PDF: ReportLab preferred; robust fallback to FPDF; includes metrics, confusion matrix, ROC/PR curves, SHAP tables
# ============================================================

from __future__ import annotations

import os
import sys
import re
import json
import smtplib
import warnings
import importlib
import importlib.util
import inspect
from dataclasses import dataclass
from io import BytesIO
from datetime import datetime, timezone, timedelta
from email.message import EmailMessage
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)

# ============================================================
# ✅ MUST BE FIRST STREAMLIT COMMAND
# ============================================================
_PAGE_CONFIG_ERR = None
try:
    st.set_page_config(
        page_title="CyberSecurity Log Classifier",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception as e:
    _PAGE_CONFIG_ERR = str(e)

# ---- Stability knobs ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None
np.seterr(all="ignore")

# ============================================================
# GLOBAL SAFE REGEX PATCH (prevents: "global flags not at the start...")
# ============================================================
if not getattr(re, "_SAFE_COMPILE_PATCHED", False):
    _orig_compile = re.compile
    _FLAG_MAP = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE, "a": re.ASCII, "u": 0}
    _GLOBAL_INLINE_FLAGS_RE = _orig_compile(r"\(\?([aimsxau]+)\)")

    def _safe_compile(pattern, flags: int = 0):
        try:
            return _orig_compile(pattern, flags)
        except re.error as e:
            if "global flags not at the start" not in str(e):
                raise
            if not isinstance(pattern, str):
                raise
            add_flags = 0

            def _strip(m):
                nonlocal add_flags
                for ch in m.group(1):
                    add_flags |= _FLAG_MAP.get(ch, 0)
                return ""

            fixed = _GLOBAL_INLINE_FLAGS_RE.sub(_strip, pattern)
            return _orig_compile(fixed, flags | add_flags)

    re.compile = _safe_compile  # type: ignore
    re._SAFE_COMPILE_PATCHED = True  # type: ignore

# ============================================================
# Paths / sys.path
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ============================================================
# UI theme
# ============================================================
st.markdown(
    """
<style>
:root{
  --bg:#FFFFFF; --sidebar:#F6F7F9; --card:#FFFFFF; --border:#E5E7EB;
  --text:#111827; --muted:#6B7280; --ok:#16A34A; --bad:#DC2626;
}
html, body, [data-testid="stAppViewContainer"]{ background:var(--bg)!important; color:var(--text)!important; }
[data-testid="stSidebar"]{ background:var(--sidebar)!important; }
.hr{ border:0; height:1px; background:var(--border); margin:12px 0; }
.block{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:12px 16px; }
.good{ color:var(--ok)!important; font-weight:700; }
.bad{ color:var(--bad)!important; font-weight:700; }
.small{ color:var(--muted)!important; font-size:0.92rem; }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown("<h1>🛡️ CyberSecurity Log Classifier</h1>", unsafe_allow_html=True)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

if _PAGE_CONFIG_ERR:
    with st.sidebar.expander("ℹ️ Streamlit config", expanded=False):
        st.caption("set_page_config() warning (safe to ignore if multipage/app already set it):")
        st.code(_PAGE_CONFIG_ERR)

# ============================================================
# Robust import of local artifacts.py
# ============================================================
def _import_module_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def _load_local_artifacts_module(root_dir: str, app_dir: str) -> Tuple[Any, str]:
    candidates = [
        os.path.join(root_dir, "artifacts.py"),
        os.path.join(root_dir, "utils", "artifacts.py"),
        os.path.join(app_dir, "artifacts.py"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                mod = _import_module_from_file("artifacts", path)
                sys.modules["artifacts"] = mod
                return mod, f"file:{path}"
            except Exception:
                pass
    mod = importlib.import_module("artifacts")
    return mod, f"import:{getattr(mod, '__file__', 'artifacts')}"

ART, ART_SRC = _load_local_artifacts_module(ROOT_DIR, APP_DIR)
with st.sidebar.expander("🧩 Artifacts import", expanded=False):
    st.caption(f"Using artifacts module from: {ART_SRC}")

def _try_get(name: str, default=None):
    return getattr(ART, name, default)

# ============================================================
# Bundle fallback (if artifacts.load_bundle missing/broken)
# ============================================================
@dataclass
class BundleFallback:
    model_dir: str
    priors: dict
    resolver_state: dict
    bytes_priors: dict
    bad_ips: set
    feature_weights: dict
    feature_columns: list
    scaler: Any

    def load_supervised(self, ui_name: str):
        mdl = None
        cal = None
        thr = None
        candidates_model = [f"{ui_name}_model.pkl", f"{ui_name}.pkl", f"{ui_name}_classifier.pkl"]
        candidates_cal = [f"{ui_name}_calibrator.pkl", f"{ui_name}_calibrated.pkl", f"{ui_name}_calibrated_model.pkl"]
        candidates_thr = [f"optimal_threshold_{ui_name}.pkl", f"threshold_{ui_name}.pkl"]

        for fn in candidates_model:
            p = os.path.join(self.model_dir, fn)
            if os.path.exists(p):
                try:
                    mdl = joblib.load(p)
                    break
                except Exception:
                    pass

        for fn in candidates_cal:
            p = os.path.join(self.model_dir, fn)
            if os.path.exists(p):
                try:
                    cal = joblib.load(p)
                    break
                except Exception:
                    pass

        for fn in candidates_thr:
            p = os.path.join(self.model_dir, fn)
            if os.path.exists(p):
                try:
                    thr = float(joblib.load(p))
                    break
                except Exception:
                    pass

        return mdl, cal, thr

def _fallback_load_bundle(model_dir: str) -> BundleFallback:
    def _load_one(names: List[str], default):
        for nm in names:
            p = os.path.join(model_dir, nm)
            if os.path.exists(p):
                try:
                    return joblib.load(p)
                except Exception:
                    try:
                        import pickle
                        with open(p, "rb") as f:
                            return pickle.load(f)
                    except Exception:
                        continue
        return default

    priors = _load_one(["priors.pkl", "priors.joblib", "priors.pkl.gz"], {})
    resolver_state = _load_one(["resolver_state.pkl", "resolver.pkl"], {})
    bytes_priors = _load_one(["bytes_priors.pkl", "bytes_stats.pkl"], {})

    bad_ips = _load_one(["bad_ips.pkl", "bad_ips_set.pkl"], set())
    if not isinstance(bad_ips, set):
        try:
            bad_ips = set(bad_ips)
        except Exception:
            bad_ips = set()

    feature_weights = _load_one(["feature_weights.pkl"], {})
    feature_columns = _load_one(["feature_columns.pkl", "extended_feature_columns.pkl"], [])
    if not isinstance(feature_columns, list):
        feature_columns = list(feature_columns) if feature_columns is not None else []

    scaler = _load_one(["scaler.pkl"], None)

    return BundleFallback(
        model_dir=model_dir,
        priors=priors if isinstance(priors, dict) else {},
        resolver_state=resolver_state if isinstance(resolver_state, dict) else {},
        bytes_priors=bytes_priors if isinstance(bytes_priors, dict) else {},
        bad_ips=bad_ips,
        feature_weights=feature_weights if isinstance(feature_weights, dict) else {},
        feature_columns=feature_columns,
        scaler=scaler,
    )

# ============================================================
# Resolve artifacts API + safe fallbacks
# ============================================================
load_bundle = _try_get("load_bundle", None)
canonicalize_columns = _try_get("canonicalize_columns", None)
load_feature_engineering = _try_get("load_feature_engineering", None)
compute_primary_flags = _try_get("compute_primary_flags", None)
prepare_model_matrix = _try_get("prepare_model_matrix", None)
predict_proba_with_optional_calibrator = _try_get("predict_proba_with_optional_calibrator", None)
ShapEngine = _try_get("ShapEngine", None)

if canonicalize_columns is None:
    def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ren = {
            "src_ip": "client_ip", "source_ip": "client_ip", "srcclient": "client_ip", "src_client": "client_ip",
            "clientip": "client_ip", "c_ip": "client_ip",
            "dst_ip": "dest_ip", "destip": "dest_ip", "destination_ip": "dest_ip", "dst": "dest_ip",
            "ua": "user_agent", "useragent": "user_agent",
            "host": "domain", "hostname": "domain",
            "url": "full_url", "uri": "url_path", "path": "url_path",
            "referer": "referrer", "ref": "referrer",
            "proc": "process", "cmd": "command",
            "time": "timestamp", "datetime": "timestamp", "ts": "timestamp",
        }
        cols_lower = {c: str(c).strip().lower() for c in df.columns}
        mapped = {c: ren[low] for c, low in cols_lower.items() if low in ren}
        if mapped:
            df = df.rename(columns=mapped)
        return df

if prepare_model_matrix is None:
    def prepare_model_matrix(X: pd.DataFrame, scaler, feature_cols: list):
        Xdf = X.copy()
        if feature_cols:
            for c in feature_cols:
                if c not in Xdf.columns:
                    Xdf[c] = 0
            Xdf = Xdf[feature_cols].copy()
        for c in Xdf.columns:
            if not pd.api.types.is_numeric_dtype(Xdf[c]):
                Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
        Xdf = Xdf.fillna(0.0)
        if scaler is not None and hasattr(scaler, "transform"):
            try:
                Xs = scaler.transform(Xdf.to_numpy())
                return np.asarray(Xs), list(Xdf.columns), Xdf
            except Exception:
                pass
        return Xdf.to_numpy(dtype=float), list(Xdf.columns), Xdf

if predict_proba_with_optional_calibrator is None:
    def predict_proba_with_optional_calibrator(model, calibrator, X_scaled, use_calibrator: bool = True):
        m = calibrator if (use_calibrator and calibrator is not None) else model
        if m is None:
            return np.zeros(len(X_scaled), dtype=float)
        if hasattr(m, "predict_proba"):
            return m.predict_proba(X_scaled)[:, 1]
        if hasattr(m, "decision_function"):
            s = m.decision_function(X_scaled)
            return 1.0 / (1.0 + np.exp(-np.clip(s, -50, 50)))
        return m.predict(X_scaled).astype(float)

# ============================================================
# Imports from your modular files
# ============================================================
try:
    from log_parser import parse_log_line_universal as _parse_line  # type: ignore
except Exception:
    from log_parser import parse_log_universal as _parse_line  # type: ignore

import imputer as imputer_mod  # ✅ module import for patching
from imputer import ForensicImputer, PRIMARY_COLS  # type: ignore
from ipv6_primary_conditions import add_ipv6_primary_conditions  # type: ignore

# ============================================================
# ✅ PATCH imputer.sanitize_domain_series to kill "~ float" crash
# ============================================================
_IPV4_RE_SAFE = re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$")

def _sanitize_domain_series_safe(host: pd.Series) -> pd.Series:
    if host is None:
        return pd.Series(dtype=object)

    s = host.copy().astype("string")
    s = s.str.strip().str.lower()
    s = s.str.strip("[](){}<>\"' ")
    s = s.str.rstrip(".,;)]}\"'")

    s = s.str.replace(r"^https?://", "", regex=True)
    s = s.str.split("/", n=1).str[0].str.split("?", n=1).str[0].str.split("#", n=1).str[0]
    s = s.str.split("@", n=1).str[-1]
    s = s.str.split(":", n=1).str[0]

    miss = {"", "-", "--", "none", "null", "nan", "na", "n/a", "unknown", "notprovided", "not_provided", "unknown_domain", "unknown-domain"}
    miss_mask = (s.isna() | s.isin(list(miss))).fillna(True)
    s = s.mask(miss_mask, pd.NA)

    s = s.mask(s.str.match(r"^\d+(?:\.\d+)*$", na=False), pd.NA)
    s = s.mask(s.str.match(_IPV4_RE_SAFE, na=False), pd.NA)

    internal_ok = (".local", ".lan", ".internal", ".intra", ".corp", ".corp.local")
    has_dot = s.str.contains(".", regex=False, na=False).astype(bool)
    is_internal = s.str.endswith(internal_ok, na=False).astype(bool)
    ok = (has_dot | is_internal).astype(bool)

    keep = (~pd.isna(s)).to_numpy(dtype=bool)
    bad = keep & np.logical_not(ok.to_numpy(dtype=bool))
    if bad.any():
        s = s.mask(pd.Series(bad, index=s.index), pd.NA)

    out = s.astype(object)
    out = out.where(pd.notna(out), np.nan)
    return out

imputer_mod.sanitize_domain_series = _sanitize_domain_series_safe

# ============================================================
# Confidential primary conditions (from your adaptive labeling code)
# ============================================================
explicit_regex_pattern = re.compile(
    r"(?i)\b("
    r"sqlmap|nikto|acunetix|netsparker|arachni|wpscan|masscan|nmap|zmap|dirbuster|gobuster|nessus|openvas|burpsuite|"
    r"metasploit|grabber|hydra|bruteforce|cracker|hashcat|johntheripper|"
    r"powershell|cmd\.exe|wmic|mshta|cscript|wscript|regsvr32|certutil|installutil|msbuild|schtasks|rundll32|psexec|wevtutil|"
    r"dnscmd|esentutl|forfiles|makecab|expand|bitsadmin|"
    r"zeus|emotet|mirai|mozi|trickbot|dridex|qbot|agenttesla|rat|keylogger|"
    r"malware|virus|trojan|worm|spyware|ransomware|rootkit|cryptojack|coinhive|miner|"
    r"pirate|warez|keygen|crack|torrent|carding|ccdump|spam|spammer|massmailer|bulkmailer|emailharvest|emailcollector"
    r")\b"
)
critical_pattern = re.compile(
    r"(?i)\b("
    r"porn|terrorist|childporn|pedo|isis|malware|ransomware|paedo|rape|molest|prostitut|traffick|explosive|"
    r"weapon|alqaeda|neo-nazi|nazism"
    r")\b"
)

SUSP_TZ_HARD_RE = re.compile(r"\b(vpn|tor|wireguard|openvpn|proxy|tunnel|teredo|6to4|isatap|ipsec)\b", re.IGNORECASE)
BOT_UA_RE = re.compile(r"\b(bot|crawler|spider|scrapy|selenium|headless)\b", re.IGNORECASE)
AUTO_CMD_RE = re.compile(r"\b(powershell|pwsh|cmd\.exe|curl|wget|certutil|bitsadmin|mshta|rundll32|regsvr32)\b", re.IGNORECASE)

def compute_confidential_primary_flag(df: pd.DataFrame,
                                     X: Optional[pd.DataFrame] = None,
                                     odd_used: Optional[np.ndarray] = None) -> np.ndarray:
    raw = df.get("raw_log", pd.Series("", index=df.index)).fillna("").astype(str)
    ua  = df.get("user_agent", pd.Series("", index=df.index)).fillna("").astype(str)
    dom = df.get("domain", pd.Series("", index=df.index)).fillna("").astype(str)
    cmd = df.get("command", pd.Series("", index=df.index)).fillna("").astype(str)
    proc= df.get("process", pd.Series("", index=df.index)).fillna("").astype(str)

    blob = (raw + " " + ua + " " + dom + " " + cmd + " " + proc)

    hit_explicit = blob.str.contains(explicit_regex_pattern, na=False)
    hit_critical = blob.str.contains(critical_pattern, na=False)

    # high-confidence ops combo: tunneling/vpn + auth/exec signals
    hit_ops = blob.str.contains(SUSP_TZ_HARD_RE, na=False) & blob.str.contains(
        r"\b(auth|token|login|signin|admin|root|shell|exec|payload|download|upload)\b",
        regex=True, na=False
    )

    flag = (hit_explicit | hit_critical | hit_ops)

    # strengthen with ipv6 tunnel evidence if feature matrix exists
    if X is not None:
        def _col(name: str):
            if name in X.columns:
                return pd.to_numeric(X[name], errors="coerce").fillna(0).to_numpy()
            return np.zeros(len(df), dtype=float)

        ipv6_tunnel_any = (_col("ipv6_tunnel_any") > 0)
        suspicious_url  = (_col("suspicious_url") > 0)
        lolbin_ua        = (_col("malicious_lolbin_ua") > 0)
        suspicious_geo   = (_col("suspicious_geo") > 0)

        odd = (odd_used.astype(int) > 0) if odd_used is not None else np.zeros(len(df), dtype=bool)
        ip_bad = (pd.to_numeric(df.get("ip_bad_truth", 0), errors="coerce").fillna(0).to_numpy() > 0)

        # if ipv6 tunnel + any other strong signals => treat as confidential primary
        tunnel_combo = ipv6_tunnel_any & (ip_bad | suspicious_geo | suspicious_url | lolbin_ua | odd)
        flag = flag | tunnel_combo

    return flag.to_numpy(dtype=bool)

def compute_suspicious_context(df: pd.DataFrame, X: Optional[pd.DataFrame], combined_primary: np.ndarray, odd_used: np.ndarray) -> np.ndarray:
    raw = df.get("raw_log", pd.Series("", index=df.index)).fillna("").astype(str)
    ua  = df.get("user_agent", pd.Series("", index=df.index)).fillna("").astype(str)
    cmd = df.get("command", pd.Series("", index=df.index)).fillna("").astype(str)
    proc= df.get("process", pd.Series("", index=df.index)).fillna("").astype(str)

    s = np.zeros(len(df), dtype=bool)
    s |= (combined_primary > 0)
    s |= raw.str.contains(SUSP_TZ_HARD_RE, na=False).to_numpy(dtype=bool)
    s |= ua.str.contains(BOT_UA_RE, na=False).to_numpy(dtype=bool)
    s |= cmd.str.contains(AUTO_CMD_RE, na=False).to_numpy(dtype=bool)
    s |= proc.str.contains(AUTO_CMD_RE, na=False).to_numpy(dtype=bool)
    s |= (odd_used.astype(int) > 0) & ua.str.contains(BOT_UA_RE, na=False).to_numpy(dtype=bool)

    if X is not None:
        for c in ["ipv6_tunnel_any", "whitelist_suspicious_combo", "timestamp_suspicious_tz", "malicious_lolbin_ua", "suspicious_url"]:
            if c in X.columns:
                s |= (pd.to_numeric(X[c], errors="coerce").fillna(0).to_numpy() > 0)

    return s

# ============================================================
# Timezone + wall-time parsing (more coverage)
# ============================================================
TZ_ABBR_OFFSETS_MIN = {
    "UTC": 0, "GMT": 0,
    "IST": 330, "PKT": 300, "BDT": 480, "MYT": 480, "SGT": 480, "HKT": 480, "ICT": 420,
    "CET": 60, "CEST": 120, "EET": 120, "EEST": 180, "BST": 60, "MSK": 180,
    "PST": -480, "PDT": -420, "MST": -420, "MDT": -360, "CST": -360, "CDPT": -300,
    "EST": -300, "EDT": -240,
    "JST": 540, "KST": 540,
    "AEST": 600, "AEDT": 660, "ACST": 570, "AWST": 480,
    "NZST": 720, "NZDT": 780,
}
TZ_ABBR_RE = re.compile(r"\b(" + "|".join(sorted(TZ_ABBR_OFFSETS_MIN.keys(), key=len, reverse=True)) + r")\b")

SITE_TZ_MIN = {
    "NYC": -300, "NY": -300, "LON": 0, "FRA": 60, "AMS": 60,
    "TOK": 540, "TKY": 540, "DEL": 330, "BLR": 330, "BOM": 330,
    "SIN": 480, "HKG": 480, "DXB": 240,
    "SF": -480, "SFO": -480, "LA": -480, "SEA": -480,
    "CHI": -360, "DAL": -360,
}

def infer_site_offset_min(workstation: str) -> Optional[int]:
    w = (workstation or "").upper()
    if not w:
        return None
    parts = re.split(r"[-_/\.]", w)
    for c in parts:
        if 2 <= len(c) <= 4 and c in SITE_TZ_MIN:
            return int(SITE_TZ_MIN[c])
    return None

_TS_EXTRACTORS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[ ]?[+-]\d{2}:?\d{2})\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4}\b"),
    re.compile(r"\b\d{1,2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4}\b"),
    re.compile(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4}\b"),
    re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}(?:\.\d+)?(?:\s*(?:AM|PM))?\b", re.IGNORECASE),
    re.compile(r"\b(?:audit\()?\s*(\d{9,19})(?:\.\d+)?(?:\))?\b"),
]

def extract_ts_candidate(ts_str: str, raw_line: str) -> str:
    s = (ts_str or "").strip().strip("[]")
    if s:
        return s
    r = raw_line or ""
    for rx in _TS_EXTRACTORS:
        m = rx.search(r)
        if not m:
            continue
        if m.lastindex and m.lastindex >= 1:
            return (m.group(1) or "").strip()
        return (m.group(0) or "").strip()
    return ""

def _parse_offset_token(tok: str) -> Optional[int]:
    tok = (tok or "").strip()
    if not tok:
        return None
    if tok == "Z":
        return 0
    m = re.match(r"^([+-])(\d{1,2}):?(\d{2})$", tok)
    if m:
        sign = 1 if m.group(1) == "+" else -1
        hh = int(m.group(2)); mm = int(m.group(3))
        return sign * (hh * 60 + mm)
    return None

def parse_wall_dt_and_offset(ts_str: str, raw_line: str = "", workstation: str = "") -> Tuple[datetime, int, int]:
    raw_line = raw_line or ""
    workstation = workstation or ""
    suspicious = bool(SUSP_TZ_HARD_RE.search(raw_line))

    s = extract_ts_candidate(ts_str, raw_line).strip().strip("[]")

    # epoch (10/13/16/19)
    if s.isdigit():
        try:
            n = int(s)
            if len(s) == 10:
                dt = datetime.fromtimestamp(n, tz=timezone.utc)
                return dt.replace(tzinfo=None), 0, 1
            if len(s) == 13:
                dt = datetime.fromtimestamp(n / 1000.0, tz=timezone.utc)
                return dt.replace(tzinfo=None), 0, 1
            if len(s) == 16:
                dt = datetime.fromtimestamp(n / 1_000_000.0, tz=timezone.utc)
                return dt.replace(tzinfo=None), 0, 1
            if len(s) == 19:
                dt = datetime.fromtimestamp(n / 1_000_000_000.0, tz=timezone.utc)
                return dt.replace(tzinfo=None), 0, 1
        except Exception:
            pass

    # Apache
    try:
        if re.match(r"^\d{1,2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4}$", s):
            dt = datetime.strptime(s, "%d/%b/%Y:%H:%M:%S %z")
            off = int(dt.utcoffset().total_seconds() // 60)
            return dt.replace(tzinfo=None), off, 1
    except Exception:
        pass

    # YYYY-MM-DD HH:MM:SS +0530
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4}$", s):
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S %z")
            off = int(dt.utcoffset().total_seconds() // 60)
            return dt.replace(tzinfo=None), off, 1
    except Exception:
        pass

    off_hint = None
    m_off = re.search(r"([+-]\d{2}:?\d{2})\b", s)
    if m_off:
        off_hint = _parse_offset_token(m_off.group(1))

    mz = TZ_ABBR_RE.search(raw_line)
    if mz:
        off_hint = TZ_ABBR_OFFSETS_MIN.get(mz.group(1), off_hint)

    if (off_hint is None) and (not suspicious):
        off_hint = infer_site_offset_min(workstation)

    # dateutil
    try:
        from dateutil import parser as dtparser  # type: ignore
        from dateutil.tz import tzoffset  # type: ignore
        tzinfos = {k: tzoffset(k, v * 60) for k, v in TZ_ABBR_OFFSETS_MIN.items()}
        dt = dtparser.parse(s, tzinfos=tzinfos, fuzzy=True)
        if getattr(dt, "tzinfo", None) is not None and dt.utcoffset() is not None:
            off = int(dt.utcoffset().total_seconds() // 60)
            return dt.replace(tzinfo=None), off, 1
        if off_hint is None or suspicious:
            off_hint = 0
        return dt.replace(tzinfo=None), int(off_hint), 1
    except Exception:
        pass

    # fallback formats
    fmts = [
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            if off_hint is None or suspicious:
                off_hint = 0
            return dt, int(off_hint), 1
        except Exception:
            continue

    return datetime.utcnow(), 0, 0

def compute_odd_hours_from_wall(wall_dt: datetime) -> Tuple[int, float]:
    hf = wall_dt.hour + wall_dt.minute / 60.0
    odd = 1 if (hf >= 23.0 or hf < 5.5) else 0
    return odd, hf

def compute_odd_hours_from_utc(ts_utc: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts_utc, errors="coerce", utc=True)
    hf = t.dt.hour.fillna(0).astype(float) + t.dt.minute.fillna(0).astype(float) / 60.0
    return ((hf >= 23.0) | (hf < 5.5)).astype(np.int8).to_numpy()

# ============================================================
# Streamlit-safe UI wrappers
# ============================================================
def ui_df(df: pd.DataFrame, **kwargs):
    try:
        return st.dataframe(df, use_container_width=True, **kwargs)
    except TypeError:
        return st.dataframe(df, **kwargs)

def ui_btn(label: str, **kwargs) -> bool:
    try:
        return bool(st.button(label, use_container_width=True, **kwargs))
    except TypeError:
        return bool(st.button(label, **kwargs))

def ui_dl(label: str, data: bytes, file_name: str, mime: str, **kwargs):
    try:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, use_container_width=True, **kwargs)
    except TypeError:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, **kwargs)

def _arrow(val: float, eps: float = 1e-12) -> str:
    if val > eps:
        return "↑"
    if val < -eps:
        return "↓"
    return "→"

def _fig_to_png(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def parse_index_list(text: str, n: int) -> List[int]:
    """
    Supports:
      "1,2,5"
      "1-5"
      "1, 3-7, 10"
    Returns sorted unique indices within [0, n-1].
    """
    out: List[int] = []
    t = (text or "").strip()
    if not t:
        return out
    parts = [p.strip() for p in t.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a = a.strip(); b = b.strip()
            if a.isdigit() and b.isdigit():
                lo = int(a); hi = int(b)
                if lo > hi:
                    lo, hi = hi, lo
                out.extend(list(range(lo, hi + 1)))
        else:
            if p.isdigit():
                out.append(int(p))
    out = sorted(set([i for i in out if 0 <= i < n]))
    return out

# ============================================================
# Model/artifact directory
# ============================================================
DEFAULT_MODEL_DIR_WIN = r"C:\Users\prash\Downloads\cyber_deployment\results-cyber\models_comb_saa"
DEFAULT_MODEL_DIR_REL = os.path.join(ROOT_DIR, "models_comb_saa")
ALT_MODEL_DIR_WIN = r"C:\Users\prash\Downloads\cyber_deployment\results-cyber\models_comb_sa"
ALT_MODEL_DIR_REL = os.path.join(ROOT_DIR, "models_comb_sa")

_model_dir_default = None
for cand in (DEFAULT_MODEL_DIR_WIN, DEFAULT_MODEL_DIR_REL, ALT_MODEL_DIR_WIN, ALT_MODEL_DIR_REL):
    if os.path.isdir(cand):
        _model_dir_default = cand
        break
if _model_dir_default is None:
    _model_dir_default = DEFAULT_MODEL_DIR_REL

model_dir = os.getenv("MODEL_DIR", _model_dir_default)
with st.sidebar.expander("⚙️ Advanced", expanded=False):
    model_dir = st.text_input("📦 Model directory", value=model_dir, key="adv_model_dir")
    st.caption("Timezone inferred from RAW log timestamp patterns (no secrets.toml).")

if not os.path.isdir(model_dir):
    st.error("Model directory not found. Fix it in Sidebar → Advanced.")
    st.stop()

# ============================================================
# Load bundle
# ============================================================
if callable(load_bundle):
    try:
        BUNDLE = load_bundle(model_dir)
    except Exception as e:
        st.warning(f"artifacts.load_bundle failed: {e} — using fallback loader.")
        BUNDLE = _fallback_load_bundle(model_dir)
else:
    st.warning("artifacts.load_bundle not found — using fallback loader.")
    BUNDLE = _fallback_load_bundle(model_dir)

BAD_IPS = set(getattr(BUNDLE, "bad_ips", set()) or set())
SCALER = getattr(BUNDLE, "scaler", None)
BUNDLE_FEATURE_COLS = getattr(BUNDLE, "feature_columns", []) or []
BUNDLE_FEATURE_WEIGHTS = getattr(BUNDLE, "feature_weights", {}) or {}

# Optional MoE
MOE_META_PATH = os.path.join(model_dir, "MoE_meta_model.pkl")
MOE_FEAT_PATH = os.path.join(model_dir, "MoE_meta_features.pkl")
MOE_EXP_PATH  = os.path.join(model_dir, "MoE_experts.pkl")
ISO_PATH_A    = os.path.join(model_dir, "iso_forest.pkl")
ISO_PATH_B    = os.path.join(model_dir, "isolation_forest.pkl")
ISO_PATH      = ISO_PATH_A if os.path.exists(ISO_PATH_A) else ISO_PATH_B
MOE_THR_PATH  = os.path.join(model_dir, "optimal_threshold_MoE.pkl")
MOE_AVAILABLE = all(os.path.exists(p) for p in [MOE_META_PATH, MOE_FEAT_PATH, MOE_EXP_PATH, ISO_PATH, MOE_THR_PATH])

# imputer
IMPUTER = ForensicImputer(
    priors=getattr(BUNDLE, "priors", {}) if isinstance(getattr(BUNDLE, "priors", {}), dict) else {},
    resolver_state=getattr(BUNDLE, "resolver_state", {}) if isinstance(getattr(BUNDLE, "resolver_state", {}), dict) else {},
    bytes_priors=getattr(BUNDLE, "bytes_priors", {}) if isinstance(getattr(BUNDLE, "bytes_priors", {}), dict) else {},
)

# feature engineering
try:
    if callable(load_feature_engineering):
        our_custom_feature_engineering_function = load_feature_engineering(BUNDLE)
    else:
        fe_mod = importlib.import_module("feature_engineering")
        our_custom_feature_engineering_function = getattr(fe_mod, "our_custom_feature_engineering_function")
except Exception as e:
    st.error(f"Failed to import feature_engineering. Error: {e}")
    st.stop()

# SHAP engine
SHAP_ENGINE = None
if ShapEngine is not None:
    try:
        SHAP_ENGINE = ShapEngine(BUNDLE)
    except Exception:
        SHAP_ENGINE = None

# ============================================================
# Whitelist (upload + modes)
# ============================================================
def load_whitelist_domains_csv(file_obj) -> set:
    if file_obj is None:
        return set()
    try:
        data = file_obj.getvalue()
        tmp = pd.read_csv(BytesIO(data))
        if tmp.shape[1] == 0:
            return set()
        col = tmp.columns[0]
        for c in tmp.columns:
            if str(c).strip().lower() in ("domain", "domains", "host", "hostname"):
                col = c
                break
        vals = tmp[col].dropna().astype(str).str.strip().str.lower()
        vals = vals.str.replace(r"^https?://", "", regex=True).str.split("/", n=1).str[0]
        vals = vals.str.split(":", n=1).str[0]
        return set([v for v in vals.tolist() if v and v not in ("nan", "none", "null", "-", "--")])
    except Exception:
        try:
            txt = data.decode("utf-8", errors="ignore").splitlines()
            out = set()
            for ln in txt:
                ln = (ln or "").strip()
                if not ln:
                    continue
                parts = ln.split(",")
                dom = parts[-1].strip().lower()
                dom = dom.replace("http://", "").replace("https://", "").split("/")[0].split(":")[0].strip()
                if dom and dom not in ("domain", "domains", "host", "hostname", "nan", "none", "null", "-", "--"):
                    out.add(dom)
            return out
        except Exception:
            return set()

def whitelist_hit_series(dom: pd.Series, wl: set) -> pd.Series:
    d = dom.fillna("").astype(str).str.lower().str.strip()
    if not wl:
        return pd.Series(np.zeros(len(d), dtype=np.int8), index=d.index)
    codes, uniq = pd.factorize(d, sort=False)

    def hit_one(x: str) -> int:
        if not x:
            return 0
        if x in wl:
            return 1
        parts = x.split(".")
        for k in (2, 3, 4):
            if len(parts) >= k:
                suf = ".".join(parts[-k:])
                if suf in wl:
                    return 1
        return 0

    hit_u = np.fromiter((hit_one(u) for u in uniq), dtype=np.int8, count=len(uniq))
    return pd.Series(hit_u[codes], index=d.index, dtype=np.int8)

def apply_whitelist(prob: np.ndarray,
                    dom: pd.Series,
                    suspicious_mask: np.ndarray,
                    whitelist_set: set,
                    mode: str,
                    custom_factor: Optional[float]) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    prob = np.asarray(prob, dtype=float)
    wl_hit = whitelist_hit_series(dom, whitelist_set).to_numpy(dtype=np.int8)

    if mode == "Off" or not whitelist_set:
        return np.clip(prob, 0.0, 1.0), wl_hit, 1.0, np.zeros(len(prob), dtype=bool)

    if mode == "Soft":
        factor = 0.85
    elif mode == "Medium":
        factor = 0.70
    elif mode == "Hard":
        factor = 0.50
    elif mode == "Custom":
        try:
            factor = float(custom_factor) if custom_factor is not None else 0.70
        except Exception:
            factor = 0.70
        factor = float(np.clip(factor, 0.05, 1.0))
    else:
        factor = 1.0

    mask = (wl_hit > 0) & (~np.asarray(suspicious_mask, dtype=bool))
    adj = prob.copy()
    if mask.any():
        adj[mask] = adj[mask] * factor
    return np.clip(adj, 0.0, 1.0), wl_hit, float(factor), mask

# ============================================================
# PDF generation (ReportLab preferred; fallback to FPDF)
# ============================================================
def safe_pdf_text(s: str) -> str:
    rep = {"🚨": "[MAL]", "✅": "[OK]", "↑": "^", "↓": "v", "→": "->", "—": "-", "–": "-", "…": "..."}
    s = "" if s is None else str(s)
    for k, v in rep.items():
        s = s.replace(k, v)
    return s

def build_pdf_bytes(
    results_df: pd.DataFrame,
    meta: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    shap_payloads: Optional[Dict[str, Any]] = None
) -> Tuple[bytes, str, str]:
    """
    Returns (pdf_bytes, filename, note)
    note is empty on success; otherwise contains last error.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"CyberSecurity_Report_{ts}.pdf"
    last_err = ""

    # ---------- ReportLab path ----------
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            Image as RLImage, PageBreak
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import mm

        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=landscape(A4),
            leftMargin=10 * mm, rightMargin=10 * mm, topMargin=10 * mm, bottomMargin=10 * mm
        )
        styles = getSampleStyleSheet()
        normal = ParagraphStyle("normal", parent=styles["BodyText"], fontSize=9, leading=11)
        tiny = ParagraphStyle("tiny", parent=styles["BodyText"], fontSize=7, leading=9, wordWrap="CJK")

        story: List[Any] = []
        story.append(Paragraph("CyberSecurity Log Classification Report", styles["Title"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(safe_pdf_text(meta.get("summary_line", "")), normal))
        story.append(Spacer(1, 10))

        # Metrics + CM + curves
        if metrics:
            story.append(Paragraph("Metrics (Labeled)", styles["Heading2"]))
            story.append(Spacer(1, 4))
            story.append(Paragraph(safe_pdf_text(metrics.get("line", "")), normal))
            story.append(Spacer(1, 8))

            cm = metrics.get("cm")
            if cm is not None:
                try:
                    cm_arr = np.asarray(cm, dtype=int)
                    if cm_arr.shape == (2, 2):
                        story.append(Paragraph("Confusion Matrix", styles["Heading3"]))
                        cm_tbl_data = [
                            ["", "Pred 0", "Pred 1"],
                            ["Actual 0", str(int(cm_arr[0, 0])), str(int(cm_arr[0, 1]))],
                            ["Actual 1", str(int(cm_arr[1, 0])), str(int(cm_arr[1, 1]))],
                        ]
                        cm_tbl = Table(cm_tbl_data, repeatRows=1, colWidths=[30 * mm, 30 * mm, 30 * mm])
                        cm_tbl.setStyle(TableStyle([
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]))
                        story.append(cm_tbl)
                        story.append(Spacer(1, 10))
                except Exception:
                    pass

            for key, title, w, h in [
                ("roc_png", "ROC Curve", 160 * mm, 95 * mm),
                ("pr_png",  "Precision-Recall Curve", 160 * mm, 95 * mm),
            ]:
                png = metrics.get(key)
                if png:
                    try:
                        story.append(Paragraph(title, styles["Heading3"]))
                        story.append(Spacer(1, 4))
                        story.append(RLImage(BytesIO(png), width=w, height=h))
                        story.append(Spacer(1, 12))
                    except Exception:
                        pass

        # SHAP section
        if shap_payloads:
            story.append(PageBreak())
            story.append(Paragraph("SHAP / Explanations", styles["Heading2"]))
            story.append(Spacer(1, 8))
            for key, p in shap_payloads.items():
                if not p:
                    continue
                story.append(Paragraph(safe_pdf_text(p.get("title", key)), styles["Heading3"]))
                story.append(Paragraph(safe_pdf_text(p.get("summary", "")), normal))
                story.append(Spacer(1, 6))

                png = p.get("png")
                if png:
                    try:
                        story.append(RLImage(BytesIO(png), width=250 * mm, height=120 * mm))
                        story.append(Spacer(1, 8))
                    except Exception:
                        pass

                top_rows = p.get("top_rows")
                if isinstance(top_rows, pd.DataFrame) and len(top_rows) > 0:
                    try:
                        show = top_rows.head(25).copy()
                        cols = list(show.columns)
                        data = [cols] + [[safe_pdf_text(x) for x in row] for row in show.astype(str).values.tolist()]
                        tbl = Table(data, repeatRows=1)
                        tbl.setStyle(TableStyle([
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                            ("FONTSIZE", (0, 0), (-1, -1), 7),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ]))
                        story.append(tbl)
                        story.append(Spacer(1, 12))
                    except Exception:
                        pass

        # Results table
        story.append(PageBreak())
        story.append(Paragraph("Results (RAW + Standardized)", styles["Heading2"]))
        story.append(Spacer(1, 8))

        header = [
            "idx", "prediction",
            "prob_raw", "prob_adj",
            "odd_local", "odd_utc", "odd_used",
            "primary_flag", "ipv6_primary_flag", "confidential_primary_flag",
            "override_applied", "override_reason",
            "whitelist_hit", "whitelist_mode", "whitelist_factor", "whitelist_dampened",
        ] + PRIMARY_COLS + ["raw_log"]

        data = [header]
        df_out = results_df.reset_index(drop=True)

        for i, row in df_out.iterrows():
            rec = [
                str(i),
                safe_pdf_text(str(row.get("prediction", ""))),
                "" if pd.isna(row.get("probability_raw", np.nan)) else f"{float(row.get('probability_raw')):.6f}",
                "" if pd.isna(row.get("probability", np.nan)) else f"{float(row.get('probability')):.6f}",
                str(int(row.get("odd_hours_local", 0))),
                str(int(row.get("odd_hours_utc", 0))),
                str(int(row.get("odd_hours_used", 0))),
                str(int(row.get("primary_flag", 0))),
                str(int(row.get("ipv6_primary_flag", 0))),
                str(int(row.get("confidential_primary_flag", 0))),
                str(int(row.get("override_applied", 0))),
                safe_pdf_text(str(row.get("override_reason", ""))),
                str(int(row.get("whitelist_hit", 0))),
                safe_pdf_text(str(row.get("whitelist_mode", ""))),
                "" if pd.isna(row.get("whitelist_factor", np.nan)) else f"{float(row.get('whitelist_factor')):.3f}",
                str(int(row.get("whitelist_dampened", 0))),
            ]
            for c in PRIMARY_COLS:
                rec.append(Paragraph(safe_pdf_text(str(row.get(c, ""))), tiny))
            rec.append(Paragraph(safe_pdf_text(str(row.get("raw_log", ""))[:1600]).replace("\n", " ")), tiny)
            data.append(rec)

        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(tbl)

        doc.build(story)
        return buf.getvalue(), filename, ""

    except Exception as e:
        last_err = f"ReportLab failed: {e}"

    # ---------- FPDF fallback ----------
    try:
        from fpdf import FPDF  # type: ignore

        pdf = FPDF("P", "mm", "A4")
        pdf.set_auto_page_break(True, margin=12)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.multi_cell(0, 8, safe_pdf_text("CyberSecurity Log Classification Report"), border=1, align="C")
        pdf.ln(2)
        pdf.set_font("Arial", "", 9)
        pdf.multi_cell(0, 6, safe_pdf_text(meta.get("summary_line", "")), border=1)
        pdf.ln(2)

        if metrics:
            pdf.set_font("Arial", "B", 11)
            pdf.multi_cell(0, 6, "Metrics (Labeled)", border=0)
            pdf.set_font("Arial", "", 9)
            pdf.multi_cell(0, 6, safe_pdf_text(metrics.get("line", "")), border=1)
            pdf.ln(2)

        # SHAP summaries (text-only fallback)
        if shap_payloads:
            pdf.set_font("Arial", "B", 11)
            pdf.multi_cell(0, 6, "SHAP / Explanations (text summary)", border=0)
            pdf.set_font("Arial", "", 8)
            for k, p in shap_payloads.items():
                if not p:
                    continue
                pdf.multi_cell(0, 5, safe_pdf_text(p.get("title", k)), border=1)
                pdf.multi_cell(0, 5, safe_pdf_text(p.get("summary", "")), border=1)
                pdf.ln(1)

        pdf.set_font("Arial", "B", 11)
        pdf.multi_cell(0, 6, "Results (top rows)", border=0)
        pdf.set_font("Arial", "", 7)
        for _, row in results_df.head(150).iterrows():
            line = (
                f"Pred={row.get('prediction','')} "
                f"ProbAdj={row.get('probability','')} ProbRaw={row.get('probability_raw','')} "
                f"OddUsed={row.get('odd_hours_used','')} "
                f"Override={row.get('override_applied','')} "
                f"Domain={row.get('domain','')} LogType={row.get('log_type','')}"
            )
            pdf.multi_cell(0, 4, safe_pdf_text(line), border=1)
        out = pdf.output(dest="S").encode("latin-1", errors="replace")
        return out, filename, f"{last_err} (FPDF used)."

    except Exception as e:
        last_err = f"{last_err} | FPDF failed: {e}"

    return b"", filename, last_err

# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("✅ Controls")

MODEL_UI = [
    "Decision Tree",
    "Random Forest",
    "Logistic Regression",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Ensemble (excl CatBoost)",
]
if MOE_AVAILABLE:
    MODEL_UI.append("MoE Hybrid (Supervised+ISO)")

model_choice = st.sidebar.selectbox("Select Model", MODEL_UI, key="model_choice")
use_calibrator = st.sidebar.checkbox("Use probability calibrator (if available)", value=True, key="use_calibrator")
carry_over = st.sidebar.checkbox("Carry-over last threshold when unlabeled", value=True, key="carry_over")

with st.sidebar.expander("🟩 Whitelist domains (optional)", expanded=False):
    wl_file = st.file_uploader("Upload whitelist CSV", type=["csv"], key="wl_file")
    wl_mode = st.selectbox("Whitelist mode", ["Off", "Soft", "Medium", "Hard", "Custom"], index=0, key="wl_mode")
    wl_custom = st.slider("Custom factor", 0.10, 1.00, 0.70, 0.05, key="wl_custom") if wl_mode == "Custom" else 0.70

WHITELIST = load_whitelist_domains_csv(wl_file)

with st.sidebar.expander("🔔 Alerts", expanded=False):
    enable_alerts = st.checkbox("Enable alerts", value=True, key="enable_alerts")
    alert_threshold_mode = st.selectbox("Alert threshold mode", ["Use main threshold", "Manual"], index=0, key="alert_thr_mode")
    manual_alert_thr = st.number_input("Alert threshold (prob ≥)", 0.0, 1.0, 0.5, 0.01, key="manual_alert_thr") if alert_threshold_mode == "Manual" else None

    email_enable = st.checkbox("Email alerts (optional)", value=False, key="email_enable")
    email_to = st.text_input("To (recipient email)", value="", key="email_to")
    st.caption("SMTP can be supplied via env vars or manually below.")
    smtp_host = st.text_input("SMTP host", value=os.getenv("SMTP_HOST", ""), key="smtp_host")
    smtp_port = st.number_input("SMTP port", min_value=1, max_value=65535, value=int(os.getenv("SMTP_PORT", "587") or "587"), key="smtp_port")
    smtp_user = st.text_input("SMTP username", value=os.getenv("SMTP_USER", ""), key="smtp_user")
    smtp_pass = st.text_input("SMTP password", value=os.getenv("SMTP_PASS", ""), type="password", key="smtp_pass")
    smtp_from = st.text_input("From (email)", value=os.getenv("SMTP_FROM", smtp_user), key="smtp_from")

last_thr = st.session_state.get("last_batch_threshold", None)
force_enabled = st.sidebar.checkbox("Force threshold (unlabeled)", value=False, disabled=(last_thr is None), key="force_thr")
forced_thr = st.sidebar.number_input(
    "Forced threshold", 0.0, 1.0,
    float(last_thr if last_thr is not None else 0.5),
    0.0001,
    format="%.4f",
    disabled=(not force_enabled),
    key="forced_thr",
)

# Input
input_method = st.sidebar.radio("Input Method", ["Paste Logs", "Upload Log File"], key="input_method")
logs: List[str] = []
labels_provided = False
y_true: Optional[np.ndarray] = None

if input_method == "Paste Logs":
    log_text = st.text_area("📝 Paste Logs (one per line):", height=220, key="paste_logs")
    logs = [ln.rstrip("\n") for ln in (log_text or "").split("\n") if ln.strip() != ""]
    label_text = st.text_area("Optional labels (comma-separated 0/1):", height=70, key="paste_labels")
    if label_text.strip():
        labs = [int(x.strip()) for x in label_text.split(",") if x.strip() in ("0", "1")]
        if len(labs) != len(logs):
            st.error("❌ Label count mismatch with logs.")
            st.stop()
        labels_provided = True
        y_true = np.array(labs, dtype=int)
else:
    up_file = st.file_uploader("📂 Upload log file", type=None, key="upload_log")
    lab_file = st.file_uploader("📌 Optional labels (.csv/.xlsx)", type=["csv", "xlsx"], key="upload_labels")
    if up_file:
        logs = [ln.rstrip("\n") for ln in up_file.getvalue().decode(errors="ignore").splitlines() if ln.strip() != ""]
    if lab_file:
        df_lab = pd.read_csv(lab_file) if lab_file.name.endswith(".csv") else pd.read_excel(lab_file)
        labs = df_lab.iloc[:, -1].dropna().astype(int).tolist()
        if len(labs) != len(logs):
            st.error("❌ Labels count mismatch with logs.")
            st.stop()
        labels_provided = True
        y_true = np.array(labs, dtype=int)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ============================================================
# Classify
# ============================================================
if ui_btn("🚀 Classify Logs", type="primary", key="btn_classify"):
    if not logs:
        st.error("❌ No logs provided.")
        st.stop()

    for k in [
        "df_raw", "df_imp", "results_df",
        "prob_raw", "prob_used", "predictions",
        "combined_primary", "suspicious_ctx",
        "threshold", "threshold_type",
        "used_model_name", "prob_source",
        "scaler_feats", "X_scaled",
        "shap_row", "shap_subset", "shap_all",
        "pdf_bytes", "pdf_name", "pdf_note",
    ]:
        st.session_state.pop(k, None)

    # Parse
    with st.spinner("Parsing logs …"):
        parsed: List[Dict[str, Any]] = []
        for ln in logs:
            try:
                d = _parse_line(ln)
                if d is None:
                    d = {"raw_log": ln, "log_type": "futuristic_unknown"}
            except Exception:
                d = {"raw_log": ln, "log_type": "futuristic_unknown"}
            d["raw_log"] = d.get("raw_log", ln)
            parsed.append(d)

        df_raw = pd.DataFrame(parsed)

        for c in PRIMARY_COLS:
            if c not in df_raw.columns:
                df_raw[c] = np.nan
        if "url_path" not in df_raw.columns:
            df_raw["url_path"] = np.nan
        if "raw_log" not in df_raw.columns:
            df_raw["raw_log"] = [str(x) for x in logs]

        df_raw = canonicalize_columns(df_raw)
        df_raw["timestamp_raw"] = df_raw.get("timestamp", "").fillna("").astype(str)

    # RAW timezone + odd hours
    with st.spinner("Timezone autodetect + odd-hours (RAW local wall time) …"):
        ws_s = df_raw.get("workstation", pd.Series("", index=df_raw.index)).fillna("").astype(str)
        raw_s = df_raw.get("raw_log", pd.Series("", index=df_raw.index)).fillna("").astype(str)
        ts_s = df_raw["timestamp_raw"].fillna("").astype(str)

        wall_dt, off_min, ok, odd_local, hf = [], [], [], [], []
        for i in range(len(df_raw)):
            wdt, off, okb = parse_wall_dt_and_offset(ts_s.iat[i], raw_line=raw_s.iat[i], workstation=ws_s.iat[i])
            o, hff = compute_odd_hours_from_wall(wdt)
            wall_dt.append(wdt)
            off_min.append(int(off))
            ok.append(int(okb))
            odd_local.append(int(o))
            hf.append(float(hff))

        df_raw["wall_dt"] = wall_dt
        df_raw["tz_offset_min"] = off_min
        df_raw["timestamp_parse_ok"] = ok
        df_raw["odd_hours_local"] = odd_local
        df_raw["local_hour_fraction"] = hf
        df_raw["timestamp_local_str"] = [d.strftime("%Y-%m-%d %H:%M:%S") for d in df_raw["wall_dt"]]

        utc_dt = [d - timedelta(minutes=int(m)) for d, m in zip(df_raw["wall_dt"], df_raw["tz_offset_min"])]
        df_raw["parsed_timestamp_utc"] = pd.to_datetime(pd.Series(utc_dt), utc=True, errors="coerce")
        df_raw["odd_hours_utc"] = compute_odd_hours_from_utc(df_raw["parsed_timestamp_utc"])

    # Impute
    with st.spinner("Forensic imputation (universal; no-loss) …"):
        df_imp = IMPUTER.impute_df(df_raw, fill_text="NotProvided")

        # preserve pre-impute time columns
        for c in ["wall_dt", "tz_offset_min", "timestamp_parse_ok", "odd_hours_local", "odd_hours_utc",
                  "local_hour_fraction", "timestamp_local_str", "parsed_timestamp_utc"]:
            if c in df_raw.columns and c not in df_imp.columns:
                df_imp[c] = df_raw[c].values

        # if parse failed, recompute (so alerts match user log)
        fail = pd.to_numeric(df_imp.get("timestamp_parse_ok", 0), errors="coerce").fillna(0).astype(int).eq(0)
        if fail.any():
            raw_s2 = df_imp.get("raw_log", pd.Series("", index=df_imp.index)).fillna("").astype(str)
            ws_s2 = df_imp.get("workstation", pd.Series("", index=df_imp.index)).fillna("").astype(str)
            ts_s2 = df_imp.get("timestamp", pd.Series("", index=df_imp.index)).fillna("").astype(str)

            wall_dt2, off2, ok2, odd2, hf2 = [], [], [], [], []
            for i in range(len(df_imp)):
                if not bool(fail.iat[i]):
                    wall_dt2.append(df_imp["wall_dt"].iat[i])
                    off2.append(int(df_imp["tz_offset_min"].iat[i]))
                    ok2.append(int(df_imp["timestamp_parse_ok"].iat[i]))
                    odd2.append(int(df_imp["odd_hours_local"].iat[i]))
                    hf2.append(float(df_imp["local_hour_fraction"].iat[i]))
                else:
                    wdt, offm, okb = parse_wall_dt_and_offset(ts_s2.iat[i], raw_line=raw_s2.iat[i], workstation=ws_s2.iat[i])
                    o, hff = compute_odd_hours_from_wall(wdt)
                    if okb == 0:
                        offm = 0
                    wall_dt2.append(wdt)
                    off2.append(int(offm))
                    ok2.append(int(okb))
                    odd2.append(int(o))
                    hf2.append(float(hff))

            df_imp["wall_dt"] = wall_dt2
            df_imp["tz_offset_min"] = off2
            df_imp["timestamp_parse_ok"] = ok2
            df_imp["odd_hours_local"] = odd2
            df_imp["local_hour_fraction"] = hf2
            df_imp["timestamp_local_str"] = [d.strftime("%Y-%m-%d %H:%M:%S") for d in df_imp["wall_dt"]]
            utc_dt2 = [d - timedelta(minutes=int(m)) for d, m in zip(df_imp["wall_dt"], df_imp["tz_offset_min"])]
            df_imp["parsed_timestamp_utc"] = pd.to_datetime(pd.Series(utc_dt2), utc=True, errors="coerce")
            df_imp["odd_hours_utc"] = compute_odd_hours_from_utc(df_imp["parsed_timestamp_utc"])

    # IPv6 primary conditions
    with st.spinner("IPv6 primary conditions …"):
        df_imp = add_ipv6_primary_conditions(df_imp, raw_col="raw_log")
        df_imp["ipv6_primary_flag"] = pd.to_numeric(df_imp.get("ipv6_primary_flag", 0), errors="coerce").fillna(0).astype(np.int8)

    # artifacts primary flags + basic confidential regex primary
    with st.spinner("Primary override flags …"):
        if callable(compute_primary_flags):
            try:
                df_imp["primary_flag"] = compute_primary_flags(df_imp, BUNDLE).astype(np.int8)
            except Exception:
                df_imp["primary_flag"] = np.int8(0)
        else:
            df_imp["primary_flag"] = np.int8(0)

    # prepare truth helpers for later conditions
    cip = df_imp.get("client_ip", pd.Series("", index=df_imp.index)).fillna("").astype(str).str.replace("[.]", ".", regex=False)
    dip = df_imp.get("dest_ip", pd.Series("", index=df_imp.index)).fillna("").astype(str).str.replace("[.]", ".", regex=False)
    df_imp["ip_bad_truth"] = ((cip.isin(BAD_IPS)) | (dip.isin(BAD_IPS))).astype(np.int8)

    def _is_private(ip: str) -> int:
        try:
            import ipaddress
            if not ip:
                return 0
            x = ipaddress.ip_address(ip)
            return 1 if (x.is_private or x.is_loopback or x.is_link_local) else 0
        except Exception:
            return 0

    df_imp["ip_private_truth"] = (cip.map(_is_private) | dip.map(_is_private)).astype(np.int8)

    # Feature engineering (force LOCAL wall time timestamp string)
    with st.spinner("Feature engineering …"):
        df_fe = df_imp.copy()
        df_fe["timestamp"] = df_imp.get("timestamp_local_str", df_imp.get("timestamp", "")).astype(str)
        if "raw" not in df_fe.columns:
            df_fe["raw"] = df_fe.get("raw_log", "").astype(str)
        X = our_custom_feature_engineering_function(df_fe).copy()
        X.columns = X.columns.astype(str).str.replace(" ", "_")

    # Decide odd_hours_used (local unless suspicious raw hints; then UTC)
    raw_hint_susp = df_imp.get("raw_log", "").fillna("").astype(str).str.contains(SUSP_TZ_HARD_RE, na=False).to_numpy(dtype=bool)
    ua_hint_bot = df_imp.get("user_agent", "").fillna("").astype(str).str.contains(BOT_UA_RE, na=False).to_numpy(dtype=bool)
    cmd_hint_auto = df_imp.get("command", "").fillna("").astype(str).str.contains(AUTO_CMD_RE, na=False).to_numpy(dtype=bool)
    prelim_susp = raw_hint_susp | ua_hint_bot | cmd_hint_auto | (df_imp["ipv6_primary_flag"].astype(int).to_numpy() > 0)

    odd_local_arr = pd.to_numeric(df_imp.get("odd_hours_local", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int8)
    odd_utc_arr = pd.to_numeric(df_imp.get("odd_hours_utc", 0), errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int8)
    odd_used = np.where(prelim_susp, odd_utc_arr, odd_local_arr).astype(np.int8)

    # inject odd_hours into X (so model/SHAP reflect correct wall-time policy)
    if "odd_hours" in X.columns:
        X["odd_hours"] = odd_used.astype(int)

    # Compute confidential primary with access to X + odd_used
    df_imp["confidential_primary_flag"] = compute_confidential_primary_flag(df_imp, X=X, odd_used=odd_used).astype(np.int8)

    combined_primary = (
        df_imp["primary_flag"].astype(int).to_numpy()
        | df_imp["ipv6_primary_flag"].astype(int).to_numpy()
        | df_imp["confidential_primary_flag"].astype(int).to_numpy()
    ).astype(int)

    # suspicious context (for whitelist + odd_hours_used final)
    suspicious_ctx = compute_suspicious_context(df_imp, X, combined_primary, odd_used)

    # final odd_used should follow "UTC only if suspicious"
    odd_used = np.where(suspicious_ctx, odd_utc_arr, odd_local_arr).astype(np.int8)
    df_imp["odd_hours_used"] = odd_used.astype(np.int8)

    # re-inject odd_hours for final scoring if present
    if "odd_hours" in X.columns:
        X["odd_hours"] = odd_used.astype(int)

    # prepare model matrix
    with st.spinner("Scoring …"):
        X_scaled, scaler_feats, _ = prepare_model_matrix(X, getattr(BUNDLE, "scaler", None), getattr(BUNDLE, "feature_columns", []) or [])

        prob_raw = None
        thr_saved = None
        used_model_name = None
        prob_source = None

        # MoE scoring
        if model_choice == "MoE Hybrid (Supervised+ISO)" and MOE_AVAILABLE:
            try:
                moe_meta = joblib.load(MOE_META_PATH)
                moe_cols = joblib.load(MOE_FEAT_PATH)
                moe_exps = joblib.load(MOE_EXP_PATH)
                iso = joblib.load(ISO_PATH)
                thr_saved = float(joblib.load(MOE_THR_PATH))

                X_df_scaled = pd.DataFrame(np.asarray(X_scaled), columns=list(scaler_feats), index=df_imp.index)

                parts = []
                for nm in moe_exps:
                    mdl, cal, _thr = BUNDLE.load_supervised(nm)
                    p = predict_proba_with_optional_calibrator(mdl, cal, X_scaled, use_calibrator=True)
                    parts.append(np.asarray(p, dtype=np.float32).reshape(-1, 1))

                s = iso.decision_function(X_df_scaled).astype(np.float64)
                a = -s
                a = (a - np.min(a)) / (np.ptp(a) + 1e-9)
                parts.append(a.astype(np.float32).reshape(-1, 1))

                for kf in ["ipv6_tunnel_any", "whitelist_suspicious_combo", "timestamp_suspicious_tz", "odd_hours"]:
                    if kf in moe_cols:
                        v = pd.to_numeric(X_df_scaled.get(kf, 0), errors="coerce").fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1)
                        parts.append(v)

                M = np.hstack(parts).astype(np.float32)
                pmeta = moe_meta.predict_proba(M)[:, 1]
                prob_raw = np.clip(pmeta, 0.0, 1.0)
                used_model_name = "MoE Hybrid"
                prob_source = "MoE(meta)+ISO"
            except Exception as e:
                st.warning(f"MoE scoring failed; falling back. ({e})")
                prob_raw = None
                thr_saved = None

        # Standard models / ensemble
        if prob_raw is None:
            def _load(ui_name: str):
                return BUNDLE.load_supervised(ui_name) if hasattr(BUNDLE, "load_supervised") else (None, None, None)

            if model_choice == "Ensemble (excl CatBoost)":
                order = ["LightGBM", "XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"]
                probs = []
                thrs = []
                for nm in order:
                    mdl, cal, thr = _load(nm)
                    if mdl is None:
                        continue
                    try:
                        p = predict_proba_with_optional_calibrator(mdl, cal, X_scaled, use_calibrator=use_calibrator)
                        probs.append(p)
                        if thr is not None:
                            thrs.append(float(thr))
                    except Exception:
                        continue

                if probs:
                    prob_raw = np.mean(np.vstack(probs), axis=0)
                    thr_saved = float(np.mean(thrs)) if thrs else None
                    used_model_name = "Ensemble(excl CatBoost)"
                    prob_source = "Ensemble mean probability"
            else:
                mdl, cal, thr = _load(model_choice)
                if mdl is not None:
                    try:
                        prob_raw = predict_proba_with_optional_calibrator(mdl, cal, X_scaled, use_calibrator=use_calibrator)
                        thr_saved = float(thr) if thr is not None else None
                        used_model_name = model_choice
                        prob_source = "Model probability"
                    except Exception:
                        prob_raw = None

        # Fallback score
        if prob_raw is None:
            fw = getattr(BUNDLE, "feature_weights", {}) or {}
            cols = [c for c in fw.keys() if c in X.columns]
            if cols:
                w = np.array([float(fw[c]) for c in cols], dtype=float)
                ss = (X[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy() @ w).astype(float)
                ss0 = ss - np.nanmin(ss)
                prob_raw = ss0 / (np.nanmax(ss0) + 1e-9) if np.nanmax(ss0) > 0 else np.zeros(len(df_imp), dtype=float)
            else:
                prob_raw = np.zeros(len(df_imp), dtype=float)
            used_model_name = "Fallback"
            prob_source = "Fallback(feature_weights)"
            thr_saved = None

    # Whitelist adjust ONLY if not suspicious context
    prob_adj, wl_hit, wl_factor, wl_damp = apply_whitelist(
        prob_raw,
        df_imp.get("domain", pd.Series("", index=df_imp.index)),
        suspicious_mask=suspicious_ctx,
        whitelist_set=WHITELIST,
        mode=wl_mode,
        custom_factor=(wl_custom if wl_mode == "Custom" else None),
    )
    df_imp["whitelist_hit"] = wl_hit.astype(np.int8)
    df_imp["whitelist_mode"] = wl_mode
    df_imp["whitelist_factor"] = float(wl_factor)
    df_imp["whitelist_dampened"] = wl_damp.astype(np.int8)
    df_imp["probability_raw"] = np.asarray(prob_raw, dtype=float)
    df_imp["probability"] = np.asarray(prob_adj, dtype=float)

    # Threshold selection
    threshold = None
    threshold_type = "Saved/Default"
    prev_thr = st.session_state.get("last_batch_threshold", None)

    if labels_provided and y_true is not None and len(np.unique(y_true)) == 2:
        try:
            prec_c, rec_c, th = precision_recall_curve(y_true, prob_adj)
            th = np.append(th, 1.0)
            f1c = (2 * prec_c * rec_c) / (prec_c + rec_c + 1e-9)
            threshold = float(th[np.argmax(f1c)])
            threshold_type = "Batch(F1)"
        except Exception:
            threshold = float(thr_saved) if thr_saved is not None else 0.5
            threshold_type = "Saved/Default"
    else:
        if force_enabled and prev_thr is not None:
            threshold = float(forced_thr)
            threshold_type = "Forced"
        elif carry_over and prev_thr is not None:
            threshold = float(prev_thr)
            threshold_type = "Carryover"
        else:
            threshold = float(thr_saved) if thr_saved is not None else 0.5
            threshold_type = "Saved/Default"

    if threshold is None:
        threshold = 0.5
    st.session_state["last_batch_threshold"] = float(threshold)

    # Apply override ALWAYS
    pred_prob = (np.asarray(prob_adj, dtype=float) >= float(threshold)).astype(int)
    pred_final = np.where(combined_primary > 0, 1, pred_prob).astype(int)

    df_imp["override_applied"] = (combined_primary > 0).astype(np.int8)
    def _reason_row(p: int, v6: int, c: int) -> str:
        r = []
        if p: r.append("artifact_primary")
        if v6: r.append("ipv6_primary")
        if c: r.append("confidential_primary")
        return "|".join(r)
    df_imp["override_reason"] = [
        _reason_row(int(p), int(v6), int(c))
        for p, v6, c in zip(df_imp["primary_flag"].astype(int), df_imp["ipv6_primary_flag"].astype(int), df_imp["confidential_primary_flag"].astype(int))
    ]

    df_imp["prediction"] = np.where(pred_final == 1, "🚨 Malicious", "✅ Non-Malicious")
    df_imp["used_model_name"] = used_model_name
    df_imp["prob_source"] = prob_source
    df_imp["threshold"] = float(threshold)
    df_imp["threshold_type"] = threshold_type
    df_imp["suspicious_context"] = suspicious_ctx.astype(np.int8)

    # Ensure odd_hours_* exist in df_imp
    if "odd_hours_local" not in df_imp.columns:
        df_imp["odd_hours_local"] = 0
    if "odd_hours_utc" not in df_imp.columns:
        df_imp["odd_hours_utc"] = 0

    # Store results
    display_cols = [
        "raw_log",
        "timestamp_local_str", "tz_offset_min",
        "odd_hours_local", "odd_hours_utc", "odd_hours_used",
        "primary_flag", "ipv6_primary_flag", "confidential_primary_flag",
        "override_applied", "override_reason",
        "suspicious_context",
        "whitelist_hit", "whitelist_mode", "whitelist_factor", "whitelist_dampened",
        "probability_raw", "probability", "prediction",
    ] + PRIMARY_COLS

    for c in display_cols:
        if c not in df_imp.columns:
            df_imp[c] = ""

    results_df = df_imp[display_cols].copy()

    st.session_state["df_raw"] = df_raw
    st.session_state["df_imp"] = df_imp
    st.session_state["results_df"] = results_df
    st.session_state["prob_raw"] = np.asarray(prob_raw, dtype=float)
    st.session_state["prob_used"] = np.asarray(prob_adj, dtype=float)
    st.session_state["predictions"] = pred_final
    st.session_state["combined_primary"] = combined_primary.astype(int)
    st.session_state["suspicious_ctx"] = suspicious_ctx.astype(bool)
    st.session_state["threshold"] = float(threshold)
    st.session_state["threshold_type"] = threshold_type
    st.session_state["used_model_name"] = used_model_name
    st.session_state["prob_source"] = prob_source
    st.session_state["scaler_feats"] = scaler_feats
    st.session_state["X_scaled"] = X_scaled

    st.success("✅ Done. Scroll for dashboard + alerts + SHAP + PDF.")

# ============================================================
# Dashboard + SHAP + PDF
# ============================================================
if "results_df" in st.session_state:
    results_df = st.session_state["results_df"]
    df_imp = st.session_state["df_imp"]
    prob_raw = np.asarray(st.session_state.get("prob_raw", np.zeros(len(results_df))), dtype=float)
    prob_used = np.asarray(st.session_state.get("prob_used", np.zeros(len(results_df))), dtype=float)
    preds = np.asarray(st.session_state.get("predictions", np.zeros(len(results_df))), dtype=int)
    combined_primary = np.asarray(st.session_state.get("combined_primary", np.zeros(len(results_df))), dtype=int)
    suspicious_ctx = np.asarray(st.session_state.get("suspicious_ctx", np.zeros(len(results_df), dtype=bool)), dtype=bool)

    thr = float(st.session_state.get("threshold", 0.5))
    thr_type = st.session_state.get("threshold_type", "Saved/Default")
    used_model_name = st.session_state.get("used_model_name", "")
    prob_source = st.session_state.get("prob_source", "")

    st.caption(f"Model: {used_model_name} · Score: {prob_source} · Threshold: {thr:.4f} ({thr_type})")

    mal = int(np.sum(preds))
    nonmal = int(len(preds) - mal)
    st.markdown(
        f"<div class='block'><b>Summary:</b> "
        f"<span class='bad'>{mal} Malicious</span> | <span class='good'>{nonmal} Non-Malicious</span></div>",
        unsafe_allow_html=True,
    )

    st.subheader("📋 RAW + Standardized (16 primary columns)")
    ui_df(results_df)

    ui_dl(
        "⬇️ Download Results CSV",
        data=results_df.to_csv(index=False).encode("utf-8", errors="ignore"),
        file_name="cyber_results.csv",
        mime="text/csv",
        key="dl_results_csv",
    )

    # Metrics payload for PDF
    metrics_payload = None
    if labels_provided and y_true is not None and len(np.unique(y_true)) == 2 and len(y_true) == len(preds):
        st.subheader("🎯 Evaluation Metrics (Labeled)")
        acc = float(accuracy_score(y_true, preds))
        f1v = float(f1_score(y_true, preds, zero_division=1))
        prec = float(precision_score(y_true, preds, zero_division=1))
        rec = float(recall_score(y_true, preds, zero_division=1))
        roc_auc_val = float(roc_auc_score(y_true, prob_used))
        pr_auc_val = float(average_precision_score(y_true, prob_used))

        st.markdown(
            f"<div class='block'>Accuracy: <b>{acc:.2%}</b> | F1: <b>{f1v:.2%}</b> | "
            f"Precision: <b>{prec:.2%}</b> | Recall: <b>{rec:.2%}</b> | "
            f"ROC-AUC: <b>{roc_auc_val:.2%}</b> | PR-AUC: <b>{pr_auc_val:.2%}</b></div>",
            unsafe_allow_html=True,
        )

        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        ui_df(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

        roc_png = None
        pr_png = None
        try:
            fpr, tpr, _ = roc_curve(y_true, prob_used)
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.plot(fpr, tpr)
            ax.plot([0, 1], [0, 1])
            ax.set_title(f"ROC Curve (AUC={roc_auc_val:.3f})")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            roc_png = _fig_to_png(fig)
        except Exception:
            roc_png = None

        try:
            p_curve, r_curve, _ = precision_recall_curve(y_true, prob_used)
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.plot(r_curve, p_curve)
            ax.set_title(f"Precision-Recall Curve (AP={pr_auc_val:.3f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            pr_png = _fig_to_png(fig)
        except Exception:
            pr_png = None

        metrics_payload = {
            "line": f"Accuracy={acc:.3f} | F1={f1v:.3f} | Precision={prec:.3f} | Recall={rec:.3f} | ROC-AUC={roc_auc_val:.3f} | PR-AUC={pr_auc_val:.3f}",
            "cm": cm.tolist(),
            "roc_png": roc_png,
            "pr_png": pr_png,
        }

    # Alerts
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("🔔 Alerts (Odd-hours uses local unless suspicious)")

    if enable_alerts:
        odd_used = pd.to_numeric(df_imp.get("odd_hours_used", df_imp.get("odd_hours_local", 0)), errors="coerce").fillna(0).astype(int).to_numpy().astype(bool)

        if alert_threshold_mode == "Manual" and manual_alert_thr is not None:
            alert_thr = float(manual_alert_thr)
            thr_src = "manual"
        else:
            alert_thr = float(thr)
            thr_src = "main_threshold"

        # Trigger: odd-hours_used AND (primary override OR predicted malicious OR prob>=alert_thr)
        cond = odd_used & (
            (combined_primary > 0) |
            (preds == 1) |
            (prob_used >= alert_thr)
        )
        idx = np.where(cond)[0].tolist()

        st.caption(f"Alert threshold: **{alert_thr:.3f}** (mode: {thr_src}) · triggered={len(idx)}")

        if not idx:
            st.info("No alerts triggered.")
            alerts_df = pd.DataFrame(columns=results_df.columns)
        else:
            alerts_df = results_df.iloc[idx].copy()
            ui_df(alerts_df)

        if email_enable and email_to.strip():
            if ui_btn("📧 Send alert email now", type="secondary", key="btn_send_email"):
                if alerts_df.empty:
                    st.warning("No alerts to email.")
                elif not (smtp_host and smtp_user and smtp_pass and smtp_from):
                    st.error("SMTP settings incomplete. Provide SMTP host/user/pass/from.")
                else:
                    try:
                        msg = EmailMessage()
                        msg["Subject"] = f"[Cyber Alerts] {len(alerts_df)} odd-hour alerts"
                        msg["From"] = smtp_from
                        msg["To"] = email_to.strip()
                        lines = [
                            f"Alerts: {len(alerts_df)}",
                            f"Generated at (UTC): {datetime.now(timezone.utc).isoformat()}",
                            "",
                            "Top alerts (idx | prob_adj | prob_raw | odd_used | local_time | domain | log_type):",
                        ]
                        for i0, row in alerts_df.head(60).reset_index(drop=True).iterrows():
                            lines.append(
                                f"- {i0} | {row.get('probability','')} | {row.get('probability_raw','')} | {row.get('odd_hours_used','')} | "
                                f"{row.get('timestamp_local_str','')} | {row.get('domain','')} | {row.get('log_type','')}"
                            )
                        msg.set_content("\n".join(lines))
                        with smtplib.SMTP(smtp_host, int(smtp_port), timeout=20) as s:
                            s.starttls()
                            s.login(smtp_user, smtp_pass)
                            s.send_message(msg)
                        st.success("Email sent.")
                    except Exception as e:
                        st.error(f"Email failed: {e}")

    # ============================================================
    # SHAP (Row / Subset Avg / ALL Avg) — robust model selection
    # ============================================================
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("🧠 SHAP (Row / Subset Avg / ALL Avg)")

    def pick_shap_model_name(choice: str) -> Optional[str]:
        tree_order = ["LightGBM", "XGBoost", "Random Forest", "Decision Tree"]
        if choice in tree_order:
            return choice
        for nm in tree_order:
            try:
                mdl, _, _ = BUNDLE.load_supervised(nm)
                if mdl is not None:
                    return nm
            except Exception:
                continue
        return None

    def compute_shap_rows(engine, mdl, X_df, row_indices: List[int], df_imp_local: pd.DataFrame, topk: int = 50):
        try:
            return engine.compute_for_rows(
                mdl,
                X_df,
                row_indices=row_indices,
                ip_bad_truth=df_imp_local.get("ip_bad_truth", None),
                ip_private_truth=df_imp_local.get("ip_private_truth", None),
                topk=topk,
                semantic_ip=True
            )
        except TypeError:
            # fallback signature
            return engine.compute_for_rows(mdl, X_df, row_indices=row_indices, topk=topk)
        except Exception as e:
            raise e

    if SHAP_ENGINE is None:
        st.info("SHAP engine unavailable (ShapEngine not loaded).")
    else:
        n = len(results_df)
        st.session_state.setdefault("sel_row", 0)
        st.session_state["sel_row"] = max(0, min(int(st.session_state["sel_row"]), max(0, n - 1)))

        if n <= 1:
            r = 0
            st.markdown("Only one row: **0**")
        else:
            c1, c2 = st.columns([3, 1])
            with c1:
                r_slider = st.slider("Row", 0, n - 1, int(st.session_state["sel_row"]), 1, key="shap_row_slider")
            with c2:
                r = int(st.number_input("Go to row", 0, n - 1, int(r_slider), 1, key="shap_row_num"))
            st.session_state["sel_row"] = r

        st.text_area("Raw log (selected)", str(results_df.iloc[r].get("raw_log", "")), height=140, key="raw_sel")

        st.session_state.setdefault("subset_text", "")
        subset_text = st.text_input("Subset rows (comma-separated, supports ranges like 3-10)", key="subset_text")

        scaler_feats = st.session_state.get("scaler_feats", [])
        X_scaled_cached = st.session_state.get("X_scaled", None)

        if X_scaled_cached is None:
            st.info("No SHAP cache yet. Run classification first.")
        else:
            X_df = pd.DataFrame(np.asarray(X_scaled_cached), columns=list(scaler_feats), index=df_imp.index)

            shap_choice = pick_shap_model_name(used_model_name or model_choice)
            if shap_choice is None:
                st.info("No SHAP-capable tree model found in model_dir (needs LightGBM/XGBoost/RF/DT).")
            else:
                mdl_shap, _, _ = BUNDLE.load_supervised(shap_choice)
                if mdl_shap is None:
                    st.info(f"SHAP model unavailable: {shap_choice} model file missing.")
                else:
                    cA, cB, cC = st.columns(3)
                    with cA:
                        do_row = ui_btn("Compute Row SHAP", type="secondary", key="btn_shap_row")
                    with cB:
                        do_subset = ui_btn("Compute Subset Avg SHAP", type="secondary", key="btn_shap_subset")
                    with cC:
                        do_all = ui_btn("Compute ALL Avg SHAP", type="secondary", key="btn_shap_all")

                    def _payload_from_shap_result(title: str, res: Dict[str, Any], which: str) -> Dict[str, Any]:
                        sv_rows = np.asarray(res["sv_rows"])
                        feat_names = list(res["feat_names"])

                        if which == "row":
                            v_dir = sv_rows[0]
                            v_mag = np.abs(sv_rows[0])
                            br = res.get("breakdown_rows", [res.get("breakdown_mean", {})])[0]
                        else:
                            v_dir = sv_rows.mean(axis=0)
                            v_mag = np.abs(sv_rows).mean(axis=0)  # ✅ stable magnitude for multi-row
                            br = res.get("breakdown_mean", {})

                        total = float(v_mag.sum()) + 1e-12
                        pct = 100.0 * (v_mag / total)
                        order = np.argsort(-pct)[: min(25, len(pct))]

                        fig, ax = plt.subplots(figsize=(10, 7))
                        labels = [f"{feat_names[i]} {_arrow(float(v_dir[i]))}" for i in order]
                        ax.barh(labels[::-1], pct[order][::-1])
                        ax.set_xlabel("Contribution % (sum=100)")
                        ax.set_title(title)
                        plt.tight_layout()
                        png = _fig_to_png(fig)

                        top_rows = pd.DataFrame({
                            "feature": [feat_names[i] for i in order],
                            "direction": [_arrow(float(v_dir[i])) for i in order],
                            "shap_mean": [float(v_dir[i]) for i in order],
                            "shap_abs_mean": [float(v_mag[i]) for i in order],
                            "contribution_pct": [float(pct[i]) for i in order],
                        })

                        summary = (
                            f"model={shap_choice} · "
                            f"pos={float(br.get('pos_pct', 0)):.2f}% · "
                            f"neg={float(br.get('neg_pct', 0)):.2f}% · "
                            f"neutral={float(br.get('neutral_pct', 0)):.2f}% · "
                            f"net={float(br.get('net_pct', 0)):.2f}%"
                        )
                        return {"title": title, "summary": summary, "png": png, "top_rows": top_rows}

                    if do_row:
                        with st.spinner(f"Computing SHAP (row) using {shap_choice}…"):
                            try:
                                res = compute_shap_rows(SHAP_ENGINE, mdl_shap, X_df, [r], df_imp, topk=60)
                                st.session_state["shap_row"] = _payload_from_shap_result(f"{shap_choice} SHAP (Row {r})", res, "row")
                            except Exception as e:
                                st.error(f"Row SHAP failed: {e}")

                    if do_subset:
                        with st.spinner(f"Computing SHAP (subset avg) using {shap_choice}…"):
                            idxs = parse_index_list(subset_text, n)
                            if not idxs:
                                st.warning("No valid subset indices.")
                            else:
                                try:
                                    res = compute_shap_rows(SHAP_ENGINE, mdl_shap, X_df, idxs, df_imp, topk=60)
                                    st.session_state["shap_subset"] = _payload_from_shap_result(f"{shap_choice} SHAP (Subset avg rows={idxs[:20]}{'...' if len(idxs)>20 else ''})", res, "mean")
                                except Exception as e:
                                    st.error(f"Subset SHAP failed: {e}")

                    if do_all:
                        with st.spinner(f"Computing SHAP (ALL avg) using {shap_choice}…"):
                            cap = 800
                            idxs = list(range(min(n, cap)))
                            try:
                                res = compute_shap_rows(SHAP_ENGINE, mdl_shap, X_df, idxs, df_imp, topk=60)
                                st.session_state["shap_all"] = _payload_from_shap_result(f"{shap_choice} SHAP (ALL avg rows=0..{len(idxs)-1})", res, "mean")
                            except Exception as e:
                                st.error(f"All-avg SHAP failed: {e}")

                    for key, label in [("shap_row", "Selected Row"), ("shap_subset", "Subset Avg"), ("shap_all", "ALL Avg")]:
                        p = st.session_state.get(key)
                        if not p:
                            continue
                        st.markdown(f"**{label}** — {p.get('summary','')}")
                        st.image(p.get("png"), caption=p.get("title", label))
                        tr = p.get("top_rows")
                        if isinstance(tr, pd.DataFrame):
                            ui_df(tr)

    # ============================================================
    # PDF Report
    # ============================================================
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("📄 PDF Report")

    include_shap_pdf = st.checkbox("Include SHAP in PDF (if computed)", value=True, key="pdf_include_shap")
    max_rows_pdf = st.number_input("Max rows in PDF", min_value=1, max_value=int(len(results_df)), value=int(len(results_df)), step=50, key="pdf_max_rows")

    if ui_btn("🧾 Generate PDF (prepare download)", type="secondary", key="btn_pdf"):
        with st.spinner("Building PDF …"):
            df_pdf = results_df.head(int(max_rows_pdf)).copy()

            shap_payloads = None
            if include_shap_pdf:
                shap_payloads = {}
                for k in ("shap_row", "shap_subset", "shap_all"):
                    p = st.session_state.get(k)
                    if p and isinstance(p, dict):
                        shap_payloads[k] = {
                            "title": p.get("title", ""),
                            "summary": p.get("summary", ""),
                            "png": p.get("png", None),
                            "top_rows": p.get("top_rows", None),
                        }
                if not shap_payloads:
                    shap_payloads = None

            meta = {
                "summary_line": (
                    f"Model={used_model_name} · Score={prob_source} · Threshold={thr:.4f} ({thr_type}) · "
                    f"Total={len(results_df)} · Malicious={int(np.sum(preds))} · Non-malicious={int(len(results_df)-np.sum(preds))} · "
                    f"WhitelistMode={wl_mode}"
                )
            }

            pdf_bytes, pdf_name, note = build_pdf_bytes(df_pdf, meta=meta, metrics=metrics_payload, shap_payloads=shap_payloads)
            st.session_state["pdf_bytes"] = pdf_bytes
            st.session_state["pdf_name"] = pdf_name
            st.session_state["pdf_note"] = note

            if not pdf_bytes:
                st.error(f"PDF generation failed. {note or 'Unknown error.'}")
            elif note:
                st.warning(note)

    if st.session_state.get("pdf_bytes"):
        ui_dl("📥 Download PDF", st.session_state["pdf_bytes"], st.session_state["pdf_name"], "application/pdf", key="dl_pdf")