from __future__ import annotations

from io import StringIO
import math
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["figure.dpi"] = 150

SINGLE_COLUMN_WIDTH = 3.25
DOUBLE_COLUMN_WIDTH = 6.75

DATASET_ORDER: List[str] = [
    "batch_bpw",
    "batch_abm",
    "cont_wat",
    "cont_but",
    "cont_ome",
    "cont_ind",
]

DATASET_LABELS: Dict[str, str] = {
    "batch_bpw": "batch bpw",
    "batch_abm": "batch abm",
    "cont_wat": "cont wat",
    "cont_but": "cont but",
    "cont_ome": "cont ome",
    "cont_ind": "cont ind",
}

DATASET_SHORT_LABELS: Dict[str, str] = {
    "batch_bpw": "batch\nbpw",
    "batch_abm": "batch\nabm",
    "cont_wat": "cont\nwat",
    "cont_but": "cont\nbut",
    "cont_ome": "cont\nome",
    "cont_ind": "cont\nind",
}

DEEP_METHODS: List[str] = [
    "AnoTrans",
    "CATCH",
    "DCdetector",
    "FEDformer",
    "GDN",
    "GGM-VAE",
    "H-PAD",
    "IGAD",
    "KAN-AD",
    "LNT",
    "LSTM-AE",
    "LSTM-P",
    "NeuTraLAD",
    "OracleAD",
    "PaAno",
    "PhysDiff",
    "RTdetector",
    "TimesNet",
]

SHALLOW_METHODS: List[str] = [
    "EIF",
    "GMM-HMM",
    "Gaussian HMM",
    "K-Means",
    "Threshold",
]

METHOD_ORDER: List[str] = DEEP_METHODS + SHALLOW_METHODS
FAMILY_BY_METHOD: Dict[str, str] = {
    **{method: "deep" for method in DEEP_METHODS},
    **{method: "shallow" for method in SHALLOW_METHODS},
}

METHOD_ALIASES: Dict[str, str] = {
    "anomaly_transformer": "AnoTrans",
    "anotrans": "AnoTrans",
    "AnoTrans": "AnoTrans",
    "catch": "CATCH",
    "CATCH": "CATCH",
    "dcdetector": "DCdetector",
    "DCdetector": "DCdetector",
    "fedformer": "FEDformer",
    "FEDformer": "FEDformer",
    "gdn": "GDN",
    "GDN": "GDN",
    "ggm_vae": "GGM-VAE",
    "ggm-vae": "GGM-VAE",
    "GGM-VAE": "GGM-VAE",
    "hpad": "H-PAD",
    "h-pad": "H-PAD",
    "H-PAD": "H-PAD",
    "igad": "IGAD",
    "IGAD": "IGAD",
    "kan_ad": "KAN-AD",
    "kan-ad": "KAN-AD",
    "KAN-AD": "KAN-AD",
    "lnt": "LNT",
    "LNT": "LNT",
    "lstm_ae": "LSTM-AE",
    "lstm-ae": "LSTM-AE",
    "LSTM-AE": "LSTM-AE",
    "lstm_p": "LSTM-P",
    "lstm-p": "LSTM-P",
    "LSTM-P": "LSTM-P",
    "neutralad": "NeuTraLAD",
    "NeuTraLAD": "NeuTraLAD",
    "oraclead": "OracleAD",
    "OracleAD": "OracleAD",
    "paano": "PaAno",
    "PaAno": "PaAno",
    "physdiff": "PhysDiff",
    "PhysDiff": "PhysDiff",
    "rtdetector": "RTdetector",
    "RTdetector": "RTdetector",
    "timesnet": "TimesNet",
    "TimesNet": "TimesNet",
    "eif": "EIF",
    "EIF": "EIF",
    "gmmhmm": "GMM-HMM",
    "gmm_hmm": "GMM-HMM",
    "GMM-HMM": "GMM-HMM",
    "hmm": "Gaussian HMM",
    "gaussian_hmm": "Gaussian HMM",
    "Gaussian HMM": "Gaussian HMM",
    "kmeans": "K-Means",
    "k-means": "K-Means",
    "K-Means": "K-Means",
    "threshold": "Threshold",
    "Threshold": "Threshold",
}

DATASET_ALIASES: Dict[str, str] = {
    "batch_bpw": "batch_bpw",
    "batch bpw": "batch_bpw",
    "batch_dist_ternary_1_butanol_2_propanol_water": "batch_bpw",
    "batch_abm": "batch_abm",
    "batch abm": "batch_abm",
    "batch_dist_ternary_acetone_1_butanol_methanol": "batch_abm",
    "cont_wat": "cont_wat",
    "cont wat": "cont_wat",
    "cont_single_component_water": "cont_wat",
    "cont_but": "cont_but",
    "cont but": "cont_but",
    "cont_binary_component_n_butanol": "cont_but",
    "cont_ome": "cont_ome",
    "cont ome": "cont_ome",
    "cont_reactive_ome": "cont_ome",
    "cont_reactive_ome_red": "cont_ome",
    "cont_reactive_ome_ext": "cont_ome",
    "cont_ind": "cont_ind",
    "cont ind": "cont_ind",
    "industry_process": "cont_ind",
    "industrial_process": "cont_ind",
}

REQUIRED_ALARM_COLUMNS = [
    "method",
    "family",
    "dataset",
    "alarm_mean",
    "alarm_std",
    "status",
]

DIAGNOSTIC_COLUMNS = ["method", "family", "dataset", "metric", "mean", "std", "status"]
SEED_COLUMNS = ["method", "family", "dataset", "seed", "alarm", "status"]


class OptionalFigureSkipped(RuntimeError):
    pass


FALLBACK_ALARM_CSV = """method,family,dataset,alarm_mean,alarm_std,status
AnoTrans,deep,batch_bpw,7.00,4.27,complete
AnoTrans,deep,batch_abm,77.66,3.05,complete
AnoTrans,deep,cont_wat,3.35,11.89,complete
AnoTrans,deep,cont_but,12.19,3.46,complete
AnoTrans,deep,cont_ome,18.75,12.18,complete
AnoTrans,deep,cont_ind,1.73,1.24,complete
CATCH,deep,batch_bpw,2.56,1.93,complete
CATCH,deep,batch_abm,10.34,8.02,complete
CATCH,deep,cont_wat,1.67,3.73,complete
CATCH,deep,cont_but,10.40,5.57,complete
CATCH,deep,cont_ome,25.93,18.66,complete
CATCH,deep,cont_ind,0.00,0.00,complete
DCdetector,deep,batch_bpw,4.27,4.74,complete
DCdetector,deep,batch_abm,29.30,18.68,complete
DCdetector,deep,cont_wat,2.79,6.24,complete
DCdetector,deep,cont_but,19.43,4.18,complete
DCdetector,deep,cont_ome,34.32,19.62,complete
DCdetector,deep,cont_ind,1.50,1.31,complete
FEDformer,deep,batch_bpw,11.65,1.54,complete
FEDformer,deep,batch_abm,78.64,3.39,complete
FEDformer,deep,cont_wat,0.00,0.00,complete
FEDformer,deep,cont_but,16.05,1.54,complete
FEDformer,deep,cont_ome,39.29,10.35,complete
FEDformer,deep,cont_ind,0.66,0.21,complete
GDN,deep,batch_bpw,11.92,1.12,complete
GDN,deep,batch_abm,71.64,9.69,complete
GDN,deep,cont_wat,0.00,0.00,complete
GDN,deep,cont_but,1.25,5.28,complete
GDN,deep,cont_ome,20.00,6.42,complete
GDN,deep,cont_ind,6.70,0.94,complete
GGM-VAE,deep,batch_bpw,15.26,0.00,complete
GGM-VAE,deep,batch_abm,79.48,0.00,complete
GGM-VAE,deep,cont_wat,7.74,1.87,complete
GGM-VAE,deep,cont_but,16.19,6.30,complete
GGM-VAE,deep,cont_ome,24.77,7.43,complete
GGM-VAE,deep,cont_ind,1.98,1.29,complete
H-PAD,deep,batch_bpw,13.83,0.00,complete
H-PAD,deep,batch_abm,80.63,0.00,complete
H-PAD,deep,cont_wat,,,missing
H-PAD,deep,cont_but,19.95,0.03,complete
H-PAD,deep,cont_ome,31.33,16.60,complete
H-PAD,deep,cont_ind,8.44,4.09,complete
IGAD,deep,batch_bpw,10.69,0.57,complete
IGAD,deep,batch_abm,84.45,3.77,complete
IGAD,deep,cont_wat,0.00,0.00,complete
IGAD,deep,cont_but,13.71,4.68,complete
IGAD,deep,cont_ome,33.02,13.03,complete
IGAD,deep,cont_ind,,,missing
KAN-AD,deep,batch_bpw,10.56,2.09,complete
KAN-AD,deep,batch_abm,26.77,12.77,complete
KAN-AD,deep,cont_wat,,,missing
KAN-AD,deep,cont_but,21.11,1.35,complete
KAN-AD,deep,cont_ome,40.00,14.14,complete
KAN-AD,deep,cont_ind,,,missing
LNT,deep,batch_bpw,10.25,0.78,complete
LNT,deep,batch_abm,80.15,0.39,complete
LNT,deep,cont_wat,0.51,2.67,complete
LNT,deep,cont_but,14.44,7.11,complete
LNT,deep,cont_ome,38.63,12.54,complete
LNT,deep,cont_ind,6.05,0.25,complete
LSTM-AE,deep,batch_bpw,12.99,0.10,complete
LSTM-AE,deep,batch_abm,79.74,6.70,complete
LSTM-AE,deep,cont_wat,2.85,2.63,complete
LSTM-AE,deep,cont_but,11.21,2.29,complete
LSTM-AE,deep,cont_ome,26.00,2.24,complete
LSTM-AE,deep,cont_ind,8.90,0.37,complete
LSTM-P,deep,batch_bpw,,,missing
LSTM-P,deep,batch_abm,,,missing
LSTM-P,deep,cont_wat,3.29,2.23,complete
LSTM-P,deep,cont_but,19.28,2.89,complete
LSTM-P,deep,cont_ome,45.58,12.99,complete
LSTM-P,deep,cont_ind,7.83,0.00,complete
NeuTraLAD,deep,batch_bpw,12.71,0.61,complete
NeuTraLAD,deep,batch_abm,60.40,15.36,complete
NeuTraLAD,deep,cont_wat,4.05,3.73,complete
NeuTraLAD,deep,cont_but,4.19,5.36,complete
NeuTraLAD,deep,cont_ome,49.00,2.23,complete
NeuTraLAD,deep,cont_ind,6.40,2.56,complete
OracleAD,deep,batch_bpw,11.14,0.71,complete
OracleAD,deep,batch_abm,79.95,0.15,complete
OracleAD,deep,cont_wat,11.68,0.33,complete
OracleAD,deep,cont_but,15.30,2.51,complete
OracleAD,deep,cont_ome,25.00,1.08,complete
OracleAD,deep,cont_ind,7.88,0.93,complete
PaAno,deep,batch_bpw,5.90,1.09,complete
PaAno,deep,batch_abm,21.00,14.69,complete
PaAno,deep,cont_wat,8.54,4.69,complete
PaAno,deep,cont_but,10.20,1.44,complete
PaAno,deep,cont_ome,14.16,16.26,complete
PaAno,deep,cont_ind,2.25,0.00,complete
PhysDiff,deep,batch_bpw,13.05,1.22,complete
PhysDiff,deep,batch_abm,80.31,0.06,complete
PhysDiff,deep,cont_wat,3.60,0.78,complete
PhysDiff,deep,cont_but,27.16,0.06,complete
PhysDiff,deep,cont_ome,50.00,0.00,complete
PhysDiff,deep,cont_ind,3.71,1.20,complete
RTdetector,deep,batch_bpw,7.77,4.03,complete
RTdetector,deep,batch_abm,79.82,0.20,complete
RTdetector,deep,cont_wat,0.00,0.00,complete
RTdetector,deep,cont_but,13.15,5.35,complete
RTdetector,deep,cont_ome,20.00,1.00,complete
RTdetector,deep,cont_ind,8.58,1.27,complete
TimesNet,deep,batch_bpw,0.00,0.00,complete
TimesNet,deep,batch_abm,15.81,21.80,complete
TimesNet,deep,cont_wat,0.00,0.00,complete
TimesNet,deep,cont_but,13.63,3.15,complete
TimesNet,deep,cont_ome,20.00,0.00,complete
TimesNet,deep,cont_ind,,,missing
EIF,shallow,batch_bpw,1.62,2.44,complete
EIF,shallow,batch_abm,1.92,2.33,complete
EIF,shallow,cont_wat,9.05,2.58,complete
EIF,shallow,cont_but,10.51,4.45,complete
EIF,shallow,cont_ome,8.45,18.24,complete
EIF,shallow,cont_ind,1.03,0.52,complete
GMM-HMM,shallow,batch_bpw,10.31,9.63,complete
GMM-HMM,shallow,batch_abm,80.67,0.29,complete
GMM-HMM,shallow,cont_wat,6.05,0.00,complete
GMM-HMM,shallow,cont_but,5.20,7.12,complete
GMM-HMM,shallow,cont_ome,48.50,1.37,complete
GMM-HMM,shallow,cont_ind,7.82,1.35,complete
Gaussian HMM,shallow,batch_bpw,11.85,6.11,complete
Gaussian HMM,shallow,batch_abm,70.56,5.68,complete
Gaussian HMM,shallow,cont_wat,2.26,4.90,complete
Gaussian HMM,shallow,cont_but,20.16,7.98,complete
Gaussian HMM,shallow,cont_ome,52.50,3.98,complete
Gaussian HMM,shallow,cont_ind,7.19,1.21,complete
K-Means,shallow,batch_bpw,12.55,0.71,complete
K-Means,shallow,batch_abm,80.84,0.02,complete
K-Means,shallow,cont_wat,,,missing
K-Means,shallow,cont_but,30.57,0.00,complete
K-Means,shallow,cont_ome,50.00,0.00,complete
K-Means,shallow,cont_ind,4.94,1.52,complete
Threshold,shallow,batch_bpw,17.75,0.00,complete
Threshold,shallow,batch_abm,75.88,0.00,complete
Threshold,shallow,cont_wat,16.16,0.00,complete
Threshold,shallow,cont_but,35.95,0.00,complete
Threshold,shallow,cont_ome,88.76,0.00,complete
Threshold,shallow,cont_ind,3.88,0.00,complete
"""

FALLBACK_DATASET_METADATA_CSV = """dataset,mode,features,train_series,test_series,train_steps,test_steps,test_anomalies,test_anomaly_pct,regime
batch_bpw,batch,18,28,63,189444,395712,77,21,lab_or_pilot
batch_abm,batch,18,8,16,30216,69558,13,24,lab_or_pilot
cont_wat,continuous,20,15,11,57284,37250,20,25,pilot
cont_but,continuous,34,7,8,2390,4082,24,41,pilot
cont_ome,continuous,20,5,3,2447,986,4,36,pilot
cont_ind,continuous,244,8,16,215841,1842436,361,17,production
"""


def repo_root(start: Optional[Path] = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for path in [current] + list(current.parents):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    raise RuntimeError("Could not locate repository root.")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> Tuple[Path, Path]:
    ensure_dir(outdir)
    pdf_path = outdir / f"{stem}.pdf"
    png_path = outdir / f"{stem}.png"
    metadata = {
        "Creator": "NoBoomBenchmark figure scripts",
        "Producer": "matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03, metadata=metadata)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return pdf_path, png_path


def canonical_method(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    text = re.sub(r"~?\\cite\{[^}]+\}", "", text).strip()
    text = text.replace("\\textsc", "").replace("{", "").replace("}", "")
    if text in METHOD_ALIASES:
        return METHOD_ALIASES[text]
    key = text.lower().replace(" ", "_").replace("-", "_")
    return METHOD_ALIASES.get(key)


def canonical_dataset(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in DATASET_ALIASES:
        return DATASET_ALIASES[text]
    key = text.lower().replace(" ", "_").replace("-", "_")
    return DATASET_ALIASES.get(key)


def method_family(method: str) -> str:
    try:
        return FAMILY_BY_METHOD[method]
    except KeyError as exc:
        raise ValueError(f"Unknown method family for {method!r}") from exc


def order_alarm_results(df: pd.DataFrame) -> pd.DataFrame:
    ordered = []
    indexed = df.set_index(["method", "dataset"], drop=False)
    for method in METHOD_ORDER:
        for dataset in DATASET_ORDER:
            if (method, dataset) in indexed.index:
                row = indexed.loc[(method, dataset)].copy()
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0].copy()
            else:
                row = pd.Series(
                    {
                        "method": method,
                        "family": method_family(method),
                        "dataset": dataset,
                        "alarm_mean": np.nan,
                        "alarm_std": np.nan,
                        "status": "missing",
                    }
                )
            ordered.append(row)
    return pd.DataFrame(ordered)[REQUIRED_ALARM_COLUMNS].reset_index(drop=True)


def validate_alarm_results(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_ALARM_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"alarm_results.csv missing columns: {missing_columns}")

    out = df[REQUIRED_ALARM_COLUMNS].copy()
    out["method"] = out["method"].map(canonical_method)
    out["dataset"] = out["dataset"].map(canonical_dataset)
    out = out[out["method"].isin(METHOD_ORDER) & out["dataset"].isin(DATASET_ORDER)].copy()
    out["family"] = out["method"].map(FAMILY_BY_METHOD)
    out["alarm_mean"] = pd.to_numeric(out["alarm_mean"], errors="coerce")
    out["alarm_std"] = pd.to_numeric(out["alarm_std"], errors="coerce")
    out["status"] = out["status"].fillna("").astype(str).str.lower().str.strip()
    out.loc[out["alarm_mean"].isna(), "status"] = "missing"
    out.loc[out["status"].eq(""), "status"] = "complete"
    out.loc[out["status"].eq("missing"), ["alarm_mean", "alarm_std"]] = np.nan
    out.loc[out["alarm_mean"].notna() & ~out["status"].eq("missing"), "status"] = "complete"

    invalid = out[out["status"].eq("missing") & out["alarm_mean"].notna()]
    if not invalid.empty:
        raise ValueError("Missing ALARM cells must keep blank/NaN values, not zeros.")
    complete_without_mean = out[out["status"].eq("complete") & out["alarm_mean"].isna()]
    if not complete_without_mean.empty:
        raise ValueError("Complete ALARM cells must contain alarm_mean values.")
    return order_alarm_results(out)


def normalize_alarm_result_file(path: Path, allow_negative: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    columns = set(df.columns)
    if set(REQUIRED_ALARM_COLUMNS).issubset(columns):
        return validate_alarm_results(df)

    if {"model", "dataset", "normalized_alarm_score_pct_mean"}.issubset(columns):
        rows = []
        for _, row in df.iterrows():
            method = canonical_method(row["model"])
            dataset = canonical_dataset(row["dataset"])
            if method is None or dataset is None:
                continue
            mean = pd.to_numeric(pd.Series([row.get("normalized_alarm_score_pct_mean")]), errors="coerce").iloc[0]
            std = np.nan
            if "normalized_alarm_score_pct_std" in df.columns:
                std = pd.to_numeric(pd.Series([row.get("normalized_alarm_score_pct_std")]), errors="coerce").iloc[0]
            status = "complete" if pd.notna(mean) else "missing"
            rows.append(
                {
                    "method": method,
                    "family": method_family(method),
                    "dataset": dataset,
                    "alarm_mean": mean,
                    "alarm_std": std,
                    "status": status,
                }
            )
        out = validate_alarm_results(pd.DataFrame(rows))
        if not allow_negative and out["alarm_mean"].dropna().lt(0).any():
            raise ValueError(f"{path} contains negative normalized ALARM values.")
        return out

    raise ValueError(f"{path} is not a recognizable ALARM result table.")


def candidate_result_files(root: Path) -> List[Path]:
    search_dirs = ["results", "outputs", "experiments", "data", "tables", "artifacts", "paper", "appendix", "docs"]
    suffixes = {".csv", ".tsv"}
    paths: List[Path] = []
    for dirname in search_dirs:
        base = root / dirname
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix.lower() in suffixes and not path.name.startswith("."):
                name = path.name.lower()
                if "alarm" in name or "result" in name or "summary" in name:
                    paths.append(path)
    return sorted(paths, key=lambda p: (0 if p.name == "alarm_results.csv" else 1, str(p)))


def find_clean_alarm_results(root: Path) -> Optional[Path]:
    for path in candidate_result_files(root):
        try:
            normalized = normalize_alarm_result_file(path, allow_negative=False)
        except Exception:
            continue
        complete_count = int(normalized["status"].eq("complete").sum())
        if complete_count >= 100:
            return path
    return None


def load_alarm_results(
    data_dir: Path,
    results_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, str]:
    data_dir = ensure_dir(data_dir)
    target = data_dir / "alarm_results.csv"

    if results_path is not None:
        source = resolve_repo_path(results_path)
        df = normalize_alarm_result_file(source, allow_negative=True)
        write_csv(df, target)
        return df, f"explicit results file: {source}"

    if target.exists() and not force_refresh:
        return validate_alarm_results(pd.read_csv(target)), f"existing {target}"

    root = repo_root()
    clean_path = find_clean_alarm_results(root)
    if clean_path is not None:
        df = normalize_alarm_result_file(clean_path, allow_negative=False)
        write_csv(df, target)
        return df, f"repository result file: {clean_path}"

    df = validate_alarm_results(pd.read_csv(StringIO(FALLBACK_ALARM_CSV)))
    write_csv(df, target)
    return df, "fallback Table 2 aggregate values"


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, na_rep="")


def load_dataset_metadata(data_dir: Path) -> Tuple[pd.DataFrame, str]:
    data_dir = ensure_dir(data_dir)
    target = data_dir / "dataset_metadata.csv"
    if target.exists():
        df = pd.read_csv(target)
        return validate_dataset_metadata(df), f"existing {target}"
    df = validate_dataset_metadata(pd.read_csv(StringIO(FALLBACK_DATASET_METADATA_CSV)))
    write_csv(df, target)
    return df, "fallback dataset metadata"


def validate_dataset_metadata(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "dataset",
        "mode",
        "features",
        "train_series",
        "test_series",
        "train_steps",
        "test_steps",
        "test_anomalies",
        "test_anomaly_pct",
        "regime",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"dataset_metadata.csv missing columns: {missing}")
    out = df[required].copy()
    out["dataset"] = out["dataset"].map(canonical_dataset)
    out = out[out["dataset"].isin(DATASET_ORDER)].copy()
    for column in ["features", "train_series", "test_series", "train_steps", "test_steps", "test_anomalies"]:
        out[column] = pd.to_numeric(out[column], errors="raise")
    out["test_anomaly_pct"] = pd.to_numeric(out["test_anomaly_pct"], errors="raise")
    out["dataset"] = pd.Categorical(out["dataset"], categories=DATASET_ORDER, ordered=True)
    return out.sort_values("dataset").reset_index(drop=True)


def parse_latex_metric_table(path: Path, metric: str) -> pd.DataFrame:
    text = path.read_text()
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "&" not in line or not line.endswith(r"\\"):
            continue
        if "diagbox" in line or "toprule" in line or "midrule" in line or "bottomrule" in line:
            continue
        method_token = line.split("&", 1)[0].strip()
        method = canonical_method(method_token)
        if method is None:
            continue
        cells = [cell.strip().rstrip("\\").strip() for cell in line.split("&")[1:]]
        for dataset, cell in zip(DATASET_ORDER, cells):
            if "missingcell" in cell:
                mean = np.nan
                std = np.nan
                status = "missing"
            else:
                match = re.search(r"\\resultcell\{([^}]*)\}\{([^}]*)\}", cell)
                if match is None:
                    match = re.search(r"\\bestcell\{([0-9.]+)\{\\pm\}([0-9.]+)\}", cell)
                if match is None:
                    continue
                mean = float(match.group(1))
                std = float(match.group(2).rstrip("."))
                status = "complete"
            rows.append(
                {
                    "method": method,
                    "family": method_family(method),
                    "dataset": dataset,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def load_metric_diagnostics(data_dir: Path, strict: bool = False) -> Tuple[pd.DataFrame, str]:
    data_dir = ensure_dir(data_dir)
    target = data_dir / "metric_diagnostics.csv"
    if target.exists():
        df = validate_metric_diagnostics(pd.read_csv(target))
        return df, f"existing {target}"

    root = repo_root()
    table_dir = root / "docs" / "neurips2026_ed" / "tables"
    metric_files = {
        "AAF": table_dir / "aaf_results.tex",
        "EDF": table_dir / "edf_results.tex",
        "LDF": table_dir / "ldf_results.tex",
    }
    if not all(path.exists() for path in metric_files.values()):
        message = "Skipping diagnostic figure: AAF/EDF/LDF data not found."
        if strict:
            raise FileNotFoundError(message)
        raise OptionalFigureSkipped(message)

    frames = [parse_latex_metric_table(path, metric) for metric, path in metric_files.items()]
    df = validate_metric_diagnostics(pd.concat(frames, ignore_index=True))
    if df.empty:
        message = "Skipping diagnostic figure: AAF/EDF/LDF data not found."
        if strict:
            raise FileNotFoundError(message)
        raise OptionalFigureSkipped(message)
    write_csv(df, target)
    return df, "extracted from existing AAF/EDF/LDF table files"


def validate_metric_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in DIAGNOSTIC_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"metric_diagnostics.csv missing columns: {missing}")
    out = df[DIAGNOSTIC_COLUMNS].copy()
    out["method"] = out["method"].map(canonical_method)
    out["dataset"] = out["dataset"].map(canonical_dataset)
    out = out[out["method"].isin(METHOD_ORDER) & out["dataset"].isin(DATASET_ORDER)].copy()
    out["family"] = out["method"].map(FAMILY_BY_METHOD)
    out["metric"] = out["metric"].astype(str).str.upper()
    out = out[out["metric"].isin(["AAF", "EDF", "LDF"])].copy()
    out["mean"] = pd.to_numeric(out["mean"], errors="coerce")
    out["std"] = pd.to_numeric(out["std"], errors="coerce")
    out["status"] = out["status"].fillna("").astype(str).str.lower().str.strip()
    out.loc[out["mean"].isna(), "status"] = "missing"
    out.loc[out["status"].eq(""), "status"] = "complete"
    out.loc[out["status"].eq("missing"), ["mean", "std"]] = np.nan
    return out.reset_index(drop=True)


def load_seed_results(data_dir: Path, strict: bool = False) -> Tuple[pd.DataFrame, str]:
    data_dir = ensure_dir(data_dir)
    target = data_dir / "seed_results.csv"
    if target.exists():
        return validate_seed_results(pd.read_csv(target)), f"existing {target}"

    root = repo_root()
    candidates = sorted((root / "docs" / "neurips2026_ed").glob("mlflow_server_table_seed_values_normalized_alarm_*.csv"))
    frames = []
    for path in candidates:
        df = pd.read_csv(path)
        if not {"model", "dataset", "seed", "normalized_alarm_score_pct"}.issubset(df.columns):
            continue
        rows = []
        for _, row in df.iterrows():
            method = canonical_method(row["model"])
            dataset = canonical_dataset(row["dataset"])
            if method is None or dataset is None:
                continue
            alarm = pd.to_numeric(pd.Series([row["normalized_alarm_score_pct"]]), errors="coerce").iloc[0]
            seed = pd.to_numeric(pd.Series([row["seed"]]), errors="coerce").iloc[0]
            if pd.isna(alarm) or pd.isna(seed):
                continue
            rows.append(
                {
                    "method": method,
                    "family": method_family(method),
                    "dataset": dataset,
                    "seed": int(seed),
                    "alarm": float(alarm),
                    "status": "complete",
                }
            )
        if rows:
            frame = pd.DataFrame(rows)
            frame["_source"] = path.name
            frames.append(frame)

    if not frames:
        message = "Skipping uncertainty figure: per-seed data not found."
        if strict:
            raise FileNotFoundError(message)
        raise OptionalFigureSkipped(message)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("_source").drop_duplicates(
        ["method", "dataset", "seed"], keep="last"
    )
    combined = validate_seed_results(combined.drop(columns=["_source"]))
    write_csv(combined, target)
    return combined, "extracted from existing per-seed normalized ALARM CSV files"


def validate_seed_results(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in SEED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"seed_results.csv missing columns: {missing}")
    out = df[SEED_COLUMNS].copy()
    out["method"] = out["method"].map(canonical_method)
    out["dataset"] = out["dataset"].map(canonical_dataset)
    out = out[out["method"].isin(METHOD_ORDER) & out["dataset"].isin(DATASET_ORDER)].copy()
    out["family"] = out["method"].map(FAMILY_BY_METHOD)
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    out["alarm"] = pd.to_numeric(out["alarm"], errors="coerce")
    out["status"] = out["status"].fillna("complete").astype(str).str.lower().str.strip()
    out = out[out["seed"].notna() & out["alarm"].notna()].copy()
    out["seed"] = out["seed"].astype(int)
    return out.reset_index(drop=True)


def completed_alarm(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"].eq("complete") & df["alarm_mean"].notna()].copy()


def best_method(df: pd.DataFrame, dataset: str, family: Optional[str] = None, exclude: Optional[Iterable[str]] = None) -> pd.Series:
    sub = completed_alarm(df)
    sub = sub[sub["dataset"].eq(dataset)]
    if family is not None:
        sub = sub[sub["family"].eq(family)]
    if exclude is not None:
        sub = sub[~sub["method"].isin(set(exclude))]
    if sub.empty:
        raise ValueError(f"No completed ALARM values for dataset={dataset!r}, family={family!r}.")
    return sub.loc[sub["alarm_mean"].idxmax()]


def threshold_row(df: pd.DataFrame, dataset: str) -> pd.Series:
    sub = completed_alarm(df)
    sub = sub[sub["dataset"].eq(dataset) & sub["method"].eq("Threshold")]
    if sub.empty:
        raise ValueError(f"Threshold missing for {dataset}.")
    return sub.iloc[0]


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)


def rounded_axis_limit(value: float, step: int = 10) -> float:
    if not math.isfinite(value) or value <= 0:
        return float(step)
    return float(int(math.ceil(value / step) * step))

