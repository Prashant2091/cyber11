# artifacts.py
# ============================================================
# CODE 3 — Artifacts + Models + Feature Engineering + SHAP Utils
# ------------------------------------------------------------
# ✅ Adds load_bundle() (backward compatible for your app import)
# ✅ Global safe-regex patch (prevents "global flags not at the start..." everywhere)
# ✅ Loads scaler/weights/priors/resolver/bytes_priors + patterns/keywords + bad_ips
# ✅ Loads supervised models + optional unsupervised models (IsolationForest/KMeans) if present
# ✅ Canonicalize synonyms into 16 primary columns for 35+ logs before imputation
# ✅ SHAP utilities: pos/neg/neutral sum to 100, arrows correct, semantic IP adjust
# ============================================================

from __future__ import annotations

import os
import re
import json
import pickle
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple, Set, Callable

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# GLOBAL SAFE REGEX PATCH (prevents: "global flags not at the start...")
# This protects:
# - regex inside this module
# - regex used by pandas .str.contains(...)
# - regex strings inside pickled artifacts that later get compiled
# ============================================================
if not getattr(re, "_SAFE_COMPILE_PATCHED", False):
    _orig_compile = re.compile
    _FLAG_MAP = {
        "i": re.IGNORECASE,
        "m": re.MULTILINE,
        "s": re.DOTALL,
        "x": re.VERBOSE,
        "a": re.ASCII,
        "u": 0,
    }
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

    re.compile = _safe_compile
    re._SAFE_COMPILE_PATCHED = True


# ------------------------------------------------------------
# Streamlit cache fallback (keeps runnable in scripts/notebooks)
# ------------------------------------------------------------
try:
    import streamlit as st  # type: ignore
    _cache_resource = st.cache_resource
except Exception:
    def _cache_resource(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func
    class _DummySt:
        cache_resource = staticmethod(_cache_resource)
    st = _DummySt()  # type: ignore


# ------------------------------------------------------------
# Globals / defaults
# ------------------------------------------------------------
MODEL_CANDIDATES = {
    "Decision Tree": ["Decision Tree", "DecisionTree", "DT"],
    "Random Forest": ["RF", "Random Forest", "RandomForest"],
    "Logistic Regression": ["Logistic Regression", "LogisticRegression", "LR"],
    "XGBoost": ["XGBoost", "XGB"],
    "LightGBM": ["LightGBM", "LGBM"],
    "CatBoost": ["CatBoost"],
}

MISS_TOKENS = {
    "", " ", "unknown", "missing", "null", "none", "nan", "na", "n/a",
    "-", "--", "notprovided", "not_provided", "unknown-domain", "unknown_domain"
}
MISS_STRS = {str(x).strip().lower() for x in MISS_TOKENS if x is not None}
BAD_STR_RE = re.compile(
    r"^(?:unknown|missing_token|null|none|nan|na|n/a|-|--|notprovided|unknown-domain|unknown_domain)\s*$",
    re.I
)

_IPV4_RE = re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
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

def is_missing_str(x: Any) -> bool:
    return str(x).strip().lower() in MISS_STRS

def _joblib_or_pickle(path: str):
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

def _as_set_of_str(x) -> Set[str]:
    if x is None:
        return set()
    if isinstance(x, set):
        return {str(v).replace("[.]", ".").strip() for v in x if str(v).strip()}
    if isinstance(x, (list, tuple)):
        return {str(v).replace("[.]", ".").strip() for v in x if str(v).strip()}
    return set()

def probs_degenerate(p: np.ndarray) -> bool:
    p = np.asarray(p, dtype=float).reshape(-1)
    return (
        p.size == 0 or np.isnan(p).any() or np.isinf(p).any()
        or np.ptp(p) < 1e-8 or len(np.unique(np.round(p, 6))) < 2
    )

# rankdata fallback-safe (no SciPy dependency)
try:
    from scipy.stats import rankdata as _rankdata  # type: ignore
except Exception:
    def _rankdata(a, method="average"):
        a = np.asarray(a, dtype=float).reshape(-1)
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(a) + 1, dtype=float)
        return ranks


# ------------------------------------------------------------
# Bundle dataclasses
# ------------------------------------------------------------
@dataclass
class ModelBundle:
    name_ui: str
    name_loaded: str
    model: Any
    calibrator: Any = None
    threshold: Optional[float] = None

@dataclass
class ArtifactsBundle:
    model_dir: str

    scaler: Any = None
    feature_weights: Dict[str, float] = None
    feature_columns: List[str] = None
    extended_feature_columns: Optional[List[str]] = None

    priors: Optional[dict] = None
    resolver_state: Optional[dict] = None
    bytes_priors: Optional[dict] = None

    bad_ips: Set[str] = None
    explicit_regex_pattern: Any = None
    malicious_browser_regex: Any = None
    critical_pattern: Any = None
    moderate_pattern: Any = None
    lolbin_ua_regex: Any = None
    keyword_artifacts: List[Any] = None

    raw_artifacts: Dict[str, Any] = None


# ------------------------------------------------------------
# Artifact loading
# ------------------------------------------------------------
@st.cache_resource
def load_bundle(model_dir: str) -> ArtifactsBundle:
    base = os.path.abspath(model_dir)
    raw: Dict[str, Any] = {}

    # load top-level pkls
    try:
        for fn in os.listdir(base):
            if fn.lower().endswith(".pkl"):
                raw[fn] = _joblib_or_pickle(os.path.join(base, fn))
    except Exception:
        pass

    forensic_dir = os.path.join(base, "forensic_artifacts")
    if os.path.isdir(forensic_dir):
        try:
            for fn in os.listdir(forensic_dir):
                p = os.path.join(forensic_dir, fn)
                if fn.lower().endswith(".pkl"):
                    raw[f"forensic::{fn}"] = _joblib_or_pickle(p)
                elif fn.lower().endswith(".json"):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            raw[f"forensic::{fn}"] = json.load(f)
                    except Exception:
                        raw[f"forensic::{fn}"] = None
        except Exception:
            pass

    scaler = raw.get("scaler.pkl", None)

    fw = raw.get("feature_weights.pkl", {}) if isinstance(raw.get("feature_weights.pkl", {}), dict) else {}
    fw = {str(k).replace(" ", "_"): float(v) for k, v in fw.items() if k is not None and v is not None and np.isfinite(float(v))}
    feature_weights = fw

    extended_cols = raw.get("extended_feature_columns.pkl", None)
    if isinstance(extended_cols, (list, tuple)):
        extended_feature_columns = [str(c) for c in extended_cols]
    else:
        extended_feature_columns = None

    if feature_weights:
        feature_columns = list(feature_weights.keys())
    elif extended_feature_columns:
        feature_columns = [str(c).replace(" ", "_") for c in extended_feature_columns]
    else:
        try:
            feature_columns = [str(c) for c in getattr(scaler, "feature_names_in_", [])] if scaler is not None else []
        except Exception:
            feature_columns = []

    # forensic artifacts (support both root + forensic folder)
    priors = raw.get("priors.pkl", None) or raw.get("forensic::priors.pkl", None)
    resolver_state = raw.get("resolver.pkl", None) or raw.get("forensic::resolver.pkl", None)
    bytes_priors = raw.get("baseline_bytes_priors.pkl", None) or raw.get("forensic::baseline_bytes_priors.pkl", None)

    bytes_dynamic = raw.get("bytes_dynamic_priors.pkl", None) or raw.get("forensic::bytes_dynamic_priors.pkl", None)
    if isinstance(bytes_priors, dict) and isinstance(bytes_dynamic, dict):
        bytes_priors = dict(bytes_priors)
        bytes_priors["dynamic"] = bytes_dynamic
    elif isinstance(bytes_dynamic, dict) and not isinstance(bytes_priors, dict):
        bytes_priors = {"global_max_in": 10_000, "global_max_out": 25_000, "dynamic": bytes_dynamic}

    bad_ips = _as_set_of_str(raw.get("bad_ips.pkl", None))

    explicit_regex_pattern = raw.get("explicit_regex_pattern.pkl", None)
    malicious_browser_regex = raw.get("malicious_browser_regex.pkl", None)
    critical_pattern = raw.get("critical_pattern.pkl", None)
    moderate_pattern = raw.get("moderate_pattern.pkl", None)
    lolbin_ua_regex = raw.get("lolbin_ua_regex.pkl", None)

    keyword_artifacts: List[Any] = []
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        kl = k.lower()
        if "keyword" in kl and kl.endswith(".pkl") and v is not None:
            keyword_artifacts.append(v)

    return ArtifactsBundle(
        model_dir=base,
        scaler=scaler,
        feature_weights=feature_weights,
        feature_columns=feature_columns,
        extended_feature_columns=extended_feature_columns,
        priors=priors if isinstance(priors, dict) else None,
        resolver_state=resolver_state if isinstance(resolver_state, dict) else None,
        bytes_priors=bytes_priors if isinstance(bytes_priors, dict) else None,
        bad_ips=bad_ips,
        explicit_regex_pattern=explicit_regex_pattern,
        malicious_browser_regex=malicious_browser_regex,
        critical_pattern=critical_pattern,
        moderate_pattern=moderate_pattern,
        lolbin_ua_regex=lolbin_ua_regex,
        keyword_artifacts=keyword_artifacts,
        raw_artifacts=raw,
    )


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------
@_cache_resource
def load_model_by_candidates(model_dir: str, ui_name: str, candidates: List[str]) -> Optional[ModelBundle]:
    base = os.path.abspath(model_dir)
    for nm in candidates:
        mp = os.path.join(base, f"{nm}_model.pkl")
        cp = os.path.join(base, f"{nm}_calibrator.pkl")
        tp = os.path.join(base, f"optimal_threshold_{nm}.pkl")
        if os.path.exists(mp):
            mdl = _joblib_or_pickle(mp)
            cal = _joblib_or_pickle(cp) if os.path.exists(cp) else None
            thr = _joblib_or_pickle(tp) if os.path.exists(tp) else None
            try:
                thr_val = float(thr) if thr is not None and np.isfinite(float(thr)) else None
            except Exception:
                thr_val = None
            return ModelBundle(name_ui=ui_name, name_loaded=nm, model=mdl, calibrator=cal, threshold=thr_val)
    return None

def load_supervised_models(model_dir: str) -> Dict[str, ModelBundle]:
    out: Dict[str, ModelBundle] = {}
    for ui_name, cand in MODEL_CANDIDATES.items():
        mb = load_model_by_candidates(model_dir, ui_name, cand)
        if mb is not None and mb.model is not None:
            out[ui_name] = mb
    return out


# ------------------------------------------------------------
# Feature engineering import + configuration
# ------------------------------------------------------------
def _import_module_by_name(name: str):
    import importlib
    return importlib.import_module(name)

def load_feature_engineering(bundle: ArtifactsBundle) -> Callable[..., pd.DataFrame]:
    mod = None
    fn = None

    # preferred
    try:
        mod = _import_module_by_name("feature_engineering")
        if hasattr(mod, "our_custom_feature_engineering_function"):
            fn = getattr(mod, "our_custom_feature_engineering_function")
    except Exception:
        mod = None
        fn = None

    # fallback
    if fn is None:
        try:
            mod = _import_module_by_name("utils.preprocess")
            if hasattr(mod, "our_custom_feature_engineering_function"):
                fn = getattr(mod, "our_custom_feature_engineering_function")
        except Exception:
            mod = None
            fn = None

    if fn is None:
        raise ImportError("Could not import feature engineering function from feature_engineering.py or utils.preprocess.")

    # Inject bad_ips if module uses it
    try:
        if mod is not None and hasattr(mod, "bad_ips") and bundle.bad_ips is not None:
            setattr(mod, "bad_ips", list(bundle.bad_ips))
    except Exception:
        pass

    return fn


# ------------------------------------------------------------
# Canonicalization utilities (35+ logs -> 16 primary cols)
# ------------------------------------------------------------
_CANON_MAP = {
    "client_ip": ["client_ip","src_client","src_client_ip","src_ip","source_ip","sourceip","ip","ip_address","src","id_orig_h"],
    "dest_ip": ["dest_ip","dst_ip","destination_ip","destinationip","dst","dest","remote_ip","server_ip","id_resp_h"],
    "timestamp": ["timestamp","time","event_time","date_time","datetime","log_time","timegenerated","time_generated"],
    "method": ["method","http_method","verb","action_method","request_method"],
    "full_url": ["full_url","url","request_url","requesturi","request_uri","uri","request"],
    "url_path": ["url_path","path","endpoint","resource","uri_path","request_path","request_uri_path"],
    "status": ["status","status_code","http_status","code","response_code"],
    "bytes_out": ["bytes_out","bytes_sent","sent","out_bytes","bytes","bytesout","response_bytes","resp_bytes"],
    "bytes_in": ["bytes_in","bytes_received","received","in_bytes","bytesin","request_bytes","req_bytes"],
    "domain": ["domain","host","hostname","sni","qname","destination_domain","dest_domain","server_name"],
    "referrer": ["referrer","referer","ref","http_referer"],
    "user_agent": ["user_agent","useragent","ua","agent","http_user_agent"],
    "username": ["username","user","principal","account","actor","requester","userprincipalname"],
    "workstation": ["workstation","device","host_name","computer","machine","hostname_device"],
    "process": ["process","process_name","image","exe","application","proc"],
    "command": ["command","commandline","cmd","cmdline","command_line","args"],
    "log_type": ["log_type","logtype","type","dataset","family","tag"],
    "raw_log": ["raw_log","raw","message","line","text","event"],
}

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_l = {str(c).lower(): c for c in df.columns}

    def get_col(names: List[str]) -> Optional[str]:
        for n in names:
            if n.lower() in cols_l:
                return cols_l[n.lower()]
        return None

    for canon, syns in _CANON_MAP.items():
        if canon not in df.columns:
            df[canon] = np.nan
        src = get_col(syns)
        if src is None or src == canon:
            continue

        if canon in ("status","bytes_in","bytes_out"):
            left = pd.to_numeric(df[canon], errors="coerce")
            right = pd.to_numeric(df[src], errors="coerce")
            mask = left.isna() & right.notna()
            if mask.any():
                df.loc[mask, canon] = right.loc[mask].to_numpy()
        else:
            left = df[canon].fillna("").astype(str).str.strip()
            right = df[src].fillna("").astype(str)
            mask = left.eq("") | left.str.lower().isin(MISS_STRS) | left.str.match(BAD_STR_RE, na=False)
            mask = mask & right.fillna("").astype(str).str.strip().ne("")
            if mask.any():
                df.loc[mask, canon] = right.loc[mask].to_numpy()

    # ensure raw_log exists
    if "raw_log" in df.columns and "raw" not in df.columns:
        df["raw"] = df["raw_log"]
    if "raw" in df.columns and "raw_log" not in df.columns:
        df["raw_log"] = df["raw"]

    return df


# ------------------------------------------------------------
# Whitelist domains (CSV upload)
# ------------------------------------------------------------
def load_whitelist_domains_csv(file_or_path) -> Set[str]:
    if file_or_path is None:
        return set()
    try:
        if isinstance(file_or_path, str):
            if not os.path.exists(file_or_path):
                return set()
            dfw = pd.read_csv(file_or_path)
        else:
            dfw = pd.read_csv(file_or_path)
    except Exception:
        return set()

    if dfw.shape[1] == 0:
        return set()

    col = None
    for c in dfw.columns:
        if str(c).strip().lower() in ("domain","domains","host","hostname"):
            col = c
            break
    if col is None:
        col = dfw.columns[0]

    vals = dfw[col].dropna().astype(str).str.strip().str.lower().tolist()
    vals = [v.replace("http://","").replace("https://","").split("/")[0] for v in vals]
    vals = [v for v in vals if v and not is_missing_str(v)]
    return set(vals)

def whitelist_factor(mode: str, custom_factor: Optional[float] = None) -> float:
    m = (mode or "Off").strip().lower()
    if m == "soft": return 0.85
    if m == "medium": return 0.70
    if m == "aggressive": return 0.50
    if m == "custom" and custom_factor is not None:
        return float(np.clip(custom_factor, 0.1, 1.0))
    return 1.0

def apply_whitelist(
    prob: np.ndarray,
    domains: pd.Series,
    primary_flags: pd.Series,
    whitelist_set: Set[str],
    mode: str = "Off",
    custom_factor: Optional[float] = None,
) -> np.ndarray:
    p = np.asarray(prob, dtype=float).reshape(-1)
    if not whitelist_set or mode is None or mode.lower() == "off":
        return p
    f = whitelist_factor(mode, custom_factor)
    dom = domains.fillna("").astype(str).str.lower()
    hit = dom.isin(whitelist_set).to_numpy()
    prim = primary_flags.fillna(0).astype(int).to_numpy()
    mask = (hit == 1) & (prim == 0)
    if mask.any():
        p2 = p.copy()
        p2[mask] = np.clip(p2[mask] * f, 0.0, 1.0)
        return p2
    return p


# ------------------------------------------------------------
# Confidential primary flag computation (keywords + regex pkls)
# ------------------------------------------------------------
def _keyword_hit_series(text: pd.Series, kw_obj) -> pd.Series:
    if kw_obj is None:
        return pd.Series(False, index=text.index)
    try:
        if hasattr(kw_obj, "pattern"):
            return text.str.contains(kw_obj, regex=True, na=False)
    except Exception:
        pass
    try:
        if isinstance(kw_obj, (list, tuple, set)) and len(kw_obj) > 0:
            kws = [str(x) for x in list(kw_obj) if str(x).strip() != ""]
            kws = kws[:5000]
            pat = re.compile("|".join(map(re.escape, kws)), flags=re.IGNORECASE)
            return text.str.contains(pat, regex=True, na=False)
    except Exception:
        pass
    return pd.Series(False, index=text.index)

def _s_contains(series: pd.Series, patt, is_regex: bool = True, case: bool = False) -> pd.Series:
    if patt is None:
        patt = r"$^"
    if hasattr(patt, "pattern"):
        return series.str.contains(patt, regex=True, na=False)
    try:
        return series.str.contains(patt, regex=is_regex, case=case, na=False)
    except Exception:
        return series.str.contains(patt, regex=is_regex, na=False)

def compute_primary_flags(df_std: pd.DataFrame, bundle: ArtifactsBundle) -> pd.Series:
    df = df_std
    raw = df.get("raw_log", pd.Series("", index=df.index)).fillna("").astype(str)

    canon_blob = pd.DataFrame({
        "client_ip": df.get("client_ip", ""),
        "timestamp": df.get("timestamp", ""),
        "method": df.get("method", ""),
        "full_url": df.get("full_url", ""),
        "status": df.get("status", ""),
        "bytes_out": df.get("bytes_out", ""),
        "referrer": df.get("referrer", ""),
        "user_agent": df.get("user_agent", ""),
        "bytes_in": df.get("bytes_in", ""),
        "domain": df.get("domain", ""),
        "dest_ip": df.get("dest_ip", ""),
        "username": df.get("username", ""),
        "workstation": df.get("workstation", ""),
        "process": df.get("process", ""),
        "command": df.get("command", ""),
        "log_type": df.get("log_type", ""),
    }, index=df.index).astype(str).agg(" ".join, axis=1)

    blob = (raw + " " + canon_blob).astype(str)

    kw_hit = pd.Series(False, index=df.index)
    if bundle.keyword_artifacts:
        for kw in bundle.keyword_artifacts:
            kw_hit = kw_hit | _keyword_hit_series(blob, kw)

    ua = df.get("user_agent", pd.Series("", index=df.index)).fillna("").astype(str)
    dom = df.get("domain", pd.Series("", index=df.index)).fillna("").astype(str)

    rule_regex = _s_contains(ua, bundle.explicit_regex_pattern, True) | _s_contains(dom, bundle.explicit_regex_pattern, True)
    rule_mal = _s_contains(ua, bundle.malicious_browser_regex or r"$^", True)
    rule_crit = _s_contains(ua, bundle.critical_pattern, True, False)
    rule_mod = _s_contains(ua, bundle.moderate_pattern, True, False)

    return (kw_hit | rule_regex | rule_mal | rule_crit | rule_mod).astype(int)


# ------------------------------------------------------------
# Model input prep + prediction
# ------------------------------------------------------------
def get_scaler_features(scaler: Any, fallback: List[str]) -> List[str]:
    try:
        feats = list(getattr(scaler, "feature_names_in_", [])) if scaler is not None else []
    except Exception:
        feats = []
    if feats:
        return [str(f) for f in feats]
    return [str(f) for f in (fallback or [])]

def prepare_model_matrix(
    X_df: pd.DataFrame,
    scaler: Any,
    fallback_feature_cols: List[str],
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    X = X_df.copy()
    X.columns = X.columns.astype(str)

    scaler_feats = get_scaler_features(scaler, fallback_feature_cols)

    for c in scaler_feats:
        cu = str(c).replace(" ", "_")
        if cu in X.columns and c not in X.columns:
            X[c] = X[cu]

    for c in scaler_feats:
        if c not in X.columns:
            X[c] = 0

    X_model = X[scaler_feats].copy()
    for c in X_model.columns:
        X_model[c] = pd.to_numeric(X_model[c], errors="coerce").fillna(0.0)

    if scaler is not None:
        X_scaled = scaler.transform(X_model)
    else:
        X_scaled = X_model.to_numpy(dtype=float)
    return np.asarray(X_scaled, dtype=float), scaler_feats, X_model

def predict_proba_with_optional_calibrator(
    mdl: Any,
    calibrator: Any,
    X_scaled: np.ndarray,
    use_calibrator: bool = True
) -> np.ndarray:
    if mdl is None:
        raise RuntimeError("Model is None")

    if hasattr(mdl, "predict_proba"):
        proba = np.asarray(mdl.predict_proba(X_scaled), dtype=float)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            idx = proba.shape[1] - 1
            try:
                cls = np.asarray(getattr(mdl, "classes_", []))
                if (cls == 1).any():
                    idx = int(np.where(cls == 1)[0][0])
            except Exception:
                pass
            p = proba[:, idx]
        else:
            p = proba.reshape(-1)
    else:
        raw = np.asarray(mdl.predict(X_scaled), dtype=float).reshape(-1)
        p = (raw - np.min(raw)) / (np.ptp(raw) + 1e-9)

    if use_calibrator and calibrator is not None:
        try:
            cp = np.asarray(calibrator.predict_proba(X_scaled), dtype=float)
            cand = cp[:, 1] if (cp.ndim == 2 and cp.shape[1] >= 2) else cp.reshape(-1)
            if (not probs_degenerate(cand)) and np.all(np.isfinite(cand)):
                p = cand
        except Exception:
            pass

    p = np.asarray(p, dtype=float).reshape(-1)

    if probs_degenerate(p):
        r = (_rankdata(p, method="average") - 1.0)
        r = r / max(1.0, float(len(r) - 1))
        p = np.asarray(r, dtype=float)

    return np.clip(p, 0.0, 1.0)


# ------------------------------------------------------------
# SHAP engine
# ------------------------------------------------------------
def _arrow(v: float, eps: float = 1e-9) -> str:
    if v > eps: return "↑"
    if v < -eps: return "↓"
    return "→"

def _safe_eps_from_vector(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float).reshape(-1)
    scale = np.nanmedian(np.abs(v))
    if not np.isfinite(scale) or scale <= 0:
        return 1e-9
    return max(1e-9, float(scale) * 1e-3)

def shap_contrib_breakdown(v: np.ndarray, eps: Optional[float] = None) -> Dict[str, float]:
    vv = np.asarray(v, dtype=float).reshape(-1)
    if eps is None:
        eps = _safe_eps_from_vector(vv)

    abs_v = np.abs(vv)
    total = float(np.nansum(abs_v))
    if not np.isfinite(total) or total <= 0:
        return {"pos_pct": 0.0, "neg_pct": 0.0, "neutral_pct": 100.0, "net_pct": 0.0}

    pos_abs = float(np.nansum(abs_v[vv > eps]))
    neg_abs = float(np.nansum(abs_v[vv < -eps]))
    neu_abs = max(0.0, total - pos_abs - neg_abs)

    pos_pct = 100.0 * pos_abs / total
    neg_pct = 100.0 * neg_abs / total
    neu_pct = 100.0 * neu_abs / total

    s = pos_pct + neg_pct + neu_pct
    diff = 100.0 - s
    buckets = np.array([pos_pct, neg_pct, neu_pct], dtype=float)
    j = int(np.argmax(buckets))
    if j == 0: pos_pct += diff
    elif j == 1: neg_pct += diff
    else: neu_pct += diff

    return {"pos_pct": float(pos_pct), "neg_pct": float(neg_pct), "neutral_pct": float(neu_pct), "net_pct": float(pos_pct - neg_pct)}

def shap_feature_table(feat_names: List[str], v: np.ndarray, topk: int = 40, eps: Optional[float] = None) -> pd.DataFrame:
    vv = np.asarray(v, dtype=float).reshape(-1)
    if eps is None:
        eps = _safe_eps_from_vector(vv)
    abs_v = np.abs(vv)
    total = float(np.nansum(abs_v)) + 1e-12
    pct = 100.0 * abs_v / total
    df = pd.DataFrame({
        "Feature": list(feat_names)[:len(vv)],
        "SHAP(signed)": vv[:len(feat_names)],
        "Contribution %": pct[:len(feat_names)],
    })
    df["Arrow"] = df["SHAP(signed)"].map(lambda x: _arrow(float(x), eps))
    df = df.sort_values("Contribution %", ascending=False)
    return df.head(int(topk)).reset_index(drop=True)

def get_model_feature_order(mdl: Any, fallback_cols: List[str]) -> List[str]:
    try:
        mf = getattr(mdl, "feature_names_in_", None)
        if mf is not None:
            mf = list(mf)
            if mf and not all(str(c).lower().startswith("column_") for c in mf):
                return [str(x) for x in mf]
    except Exception:
        pass
    try:
        if hasattr(mdl, "get_booster"):
            bn = mdl.get_booster().feature_names
            if bn:
                return [str(x) for x in bn]
    except Exception:
        pass
    return [str(x) for x in fallback_cols]

def _normalize_shap_values(sv, mdl: Any, target_class: int = 1) -> np.ndarray:
    if hasattr(sv, "values"):
        sv = sv.values
    if isinstance(sv, list):
        sv = sv[target_class] if len(sv) > target_class else sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        k = sv.shape[2]
        idx = k - 1
        try:
            cls = np.asarray(getattr(mdl, "classes_", []))
            if cls.size == k and (cls == target_class).any():
                idx = int(np.where(cls == target_class)[0][0])
        except Exception:
            pass
        sv = sv[:, :, idx]
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    return sv.astype(float, copy=False)

class ShapEngine:
    def __init__(self, bundle: ArtifactsBundle):
        self.bundle = bundle

    def _is_linear(self, mdl: Any) -> bool:
        name = type(mdl).__name__.lower()
        return ("logisticregression" in name) or hasattr(mdl, "coef_")

    def _compute_linear_contrib(self, mdl: Any, X_df: pd.DataFrame) -> np.ndarray:
        coef = np.asarray(getattr(mdl, "coef_", None))
        if coef is None or coef.size == 0:
            return np.zeros((len(X_df), X_df.shape[1]), dtype=float)
        if coef.ndim == 2 and coef.shape[0] > 1:
            idx = 1 if coef.shape[0] > 1 else 0
            coef = coef[idx:idx+1, :]
        coef = coef.reshape(1, -1)
        return X_df.to_numpy(dtype=float, copy=False) * coef

    def _compute_tree_shap(self, mdl: Any, X_df: pd.DataFrame) -> np.ndarray:
        import shap  # lazy
        try:
            expl = shap.TreeExplainer(mdl)
            sv = expl.shap_values(X_df)
        except Exception:
            expl = shap.Explainer(mdl, X_df)
            sv = expl(X_df)
        return _normalize_shap_values(sv, mdl, target_class=1)

    def _adjust_semantic_ip_shap(self, v: np.ndarray, feat_names: List[str], ip_bad_truth: int, ip_private_truth: int, eps: float) -> np.ndarray:
        vv = np.asarray(v, dtype=float).copy()

        def find_idx(names: List[str]) -> Optional[int]:
            for nm in names:
                try:
                    return feat_names.index(nm)
                except Exception:
                    continue
            return None

        i_bad = find_idx(["ip_bad_rep", "ip_bad_ip"])
        i_priv = find_idx(["ip_private"])

        if i_bad is not None:
            if ip_bad_truth >= 1:
                if vv[i_bad] < eps:
                    vv[i_bad] = eps
            else:
                if vv[i_bad] > 0:
                    vv[i_bad] = 0.0

        if i_priv is not None:
            if (ip_private_truth >= 1) and (ip_bad_truth <= 0) and (vv[i_priv] > 0):
                vv[i_priv] = 0.0

        return vv

    def compute_for_rows(
        self,
        mdl: Any,
        X_base_df: pd.DataFrame,
        row_indices: List[int],
        ip_bad_truth: Optional[pd.Series] = None,
        ip_private_truth: Optional[pd.Series] = None,
        topk: int = 40,
        semantic_ip: bool = True
    ) -> Dict[str, Any]:
        row_indices = [int(i) for i in row_indices]
        row_indices = [i for i in row_indices if 0 <= i < len(X_base_df)]
        if not row_indices:
            raise ValueError("No valid row indices for SHAP")

        feat_order = get_model_feature_order(mdl, list(X_base_df.columns))
        X_aligned = X_base_df.reindex(columns=feat_order, fill_value=0.0).copy()
        for c in X_aligned.columns:
            X_aligned[c] = pd.to_numeric(X_aligned[c], errors="coerce").fillna(0.0)

        X_sel = X_aligned.iloc[row_indices].copy()
        feat_names = list(X_sel.columns)

        if self._is_linear(mdl):
            sv = self._compute_linear_contrib(mdl, X_sel)
        else:
            sv = self._compute_tree_shap(mdl, X_sel)

        if sv.shape[1] != X_sel.shape[1]:
            m = min(sv.shape[1], X_sel.shape[1])
            sv = sv[:, :m]
            feat_names = feat_names[:m]

        adj = []
        breakdown_rows = []
        tables = []
        for j, ridx in enumerate(row_indices):
            v = sv[j].reshape(-1)
            eps = _safe_eps_from_vector(v)

            if semantic_ip and ip_bad_truth is not None and ip_private_truth is not None:
                ib = int(ip_bad_truth.iloc[ridx]) if ridx < len(ip_bad_truth) else 0
                ipr = int(ip_private_truth.iloc[ridx]) if ridx < len(ip_private_truth) else 0
                v = self._adjust_semantic_ip_shap(v, feat_names, ib, ipr, eps)

            adj.append(v)
            breakdown_rows.append(shap_contrib_breakdown(v, eps=eps))
            tables.append(shap_feature_table(feat_names, v, topk=topk, eps=eps))

        sv_adj = np.stack(adj, axis=0)

        v_mean = sv_adj.mean(axis=0)
        epsm = _safe_eps_from_vector(v_mean)
        breakdown_mean = shap_contrib_breakdown(v_mean, eps=epsm)
        table_mean = shap_feature_table(feat_names, v_mean, topk=topk, eps=epsm)

        return {
            "feat_names": feat_names,
            "sv_rows": sv_adj,
            "row_indices": row_indices,
            "breakdown_rows": breakdown_rows,
            "breakdown_mean": breakdown_mean,
            "tables_rows": tables,
            "table_mean": table_mean,
        }


# ------------------------------------------------------------
# Backward compatible "load_bundle" for your app.py
# ------------------------------------------------------------
@dataclass
class BundleFacade:
    model_dir: str
    scaler: Any
    feature_weights: Dict[str, float]
    feature_columns: List[str]
    priors: dict
    resolver_state: dict
    bytes_priors: dict
    bad_ips: Set[str]

    explicit_regex_pattern: Any = None
    malicious_browser_regex: Any = None
    critical_pattern: Any = None
    moderate_pattern: Any = None
    lolbin_ua_regex: Any = None
    keyword_artifacts: List[Any] = None
    keywords_artifact: Any = None

    models: Dict[str, ModelBundle] = None
    config: Dict[str, Any] = None

    isolation_forest: Any = None
    kmeans: Any = None
    unsup_thresholds: Dict[str, Any] = None

    raw_artifacts: Dict[str, Any] = None

    def load_supervised(self, ui_name: str):
        mb = (self.models or {}).get(ui_name)
        if mb is None:
            return None, None, None
        return mb.model, mb.calibrator, mb.threshold


@st.cache_resource
def load_bundle(model_dir: str) -> BundleFacade:
    b = load_artifacts_bundle(model_dir)
    models = load_supervised_models(model_dir)

    # config (optional)
    cfg: Dict[str, Any] = {}
    cfg_json = os.path.join(b.model_dir, "config.json")
    if os.path.exists(cfg_json):
        try:
            with open(cfg_json, "r", encoding="utf-8") as f:
                cfg.update(json.load(f) or {})
        except Exception:
            pass
    # optional config.pkl
    try:
        cpk = (b.raw_artifacts or {}).get("config.pkl", None)
        if isinstance(cpk, dict):
            cfg.update(cpk)
    except Exception:
        pass

    raw = b.raw_artifacts or {}

    # optional unsupervised models
    def _first(*names):
        for nm in names:
            if nm in raw:
                return raw[nm]
        for nm in names:
            k = f"forensic::{nm}"
            if k in raw:
                return raw[k]
        return None

    iso = _first("isolation_forest.pkl", "IsolationForest.pkl", "iforest.pkl", "iso_forest.pkl")
    km = _first("kmeans.pkl", "KMeans.pkl")

    thr = _first("unsup_thresholds.pkl", "unsupervised_thresholds.pkl", "thresholds_unsupervised.pkl", "unsup_thresholds.json", "unsupervised_thresholds.json")
    if isinstance(thr, str):
        try:
            thr = json.loads(thr)
        except Exception:
            thr = None
    if thr is None or not isinstance(thr, dict):
        thr = {}

    kw0 = (b.keyword_artifacts[0] if (b.keyword_artifacts and len(b.keyword_artifacts)) else None)

    return BundleFacade(
        model_dir=b.model_dir,
        scaler=b.scaler,
        feature_weights=b.feature_weights or {},
        feature_columns=b.feature_columns or [],
        priors=b.priors or {},
        resolver_state=b.resolver_state or {},
        bytes_priors=b.bytes_priors or {},
        bad_ips=b.bad_ips or set(),

        explicit_regex_pattern=b.explicit_regex_pattern,
        malicious_browser_regex=b.malicious_browser_regex,
        critical_pattern=b.critical_pattern,
        moderate_pattern=b.moderate_pattern,
        lolbin_ua_regex=b.lolbin_ua_regex,
        keyword_artifacts=b.keyword_artifacts or [],
        keywords_artifact=kw0,

        models=models,
        config=cfg,
        isolation_forest=iso,
        kmeans=km,
        unsup_thresholds=thr,
        raw_artifacts=raw,
    )