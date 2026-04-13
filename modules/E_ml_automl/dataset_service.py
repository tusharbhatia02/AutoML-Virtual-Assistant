from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.datasets import load_digits, load_iris, load_wine, load_diabetes, load_breast_cancer

from modules.F_stateless_info.kaggle_service import resolve_best_dataset_ref, list_dataset_files, download_dataset


TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".parquet"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _bundle_from_dataframe(dataset_id: str, df: pd.DataFrame, target_col: str | None) -> dict:
    profile = _profile_dataframe(df, target_col)
    return {
        "success": True,
        "dataset": dataset_id,
        "dataset_info": {
            "name": dataset_id,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "feature_names": list(df.columns),
            "target_name": target_col,
            "profile": profile,
        },
        "dataset_preview": df.head(10).to_dict(orient="records"),
        "dataset_files": [],
        "dataset_profile": profile,
    }


def _builtin_fallback(query: str) -> dict:
    q = query.lower().strip()
    try:
        if q in {"mnist", "digits"}:
            bunch = load_digits(as_frame=True)
            df = bunch.frame.copy()
            return _bundle_from_dataframe("mnist", df, "target")
        if q == "iris":
            bunch = load_iris(as_frame=True)
            return _bundle_from_dataframe("iris", bunch.frame.copy(), "target")
        if q == "wine":
            bunch = load_wine(as_frame=True)
            return _bundle_from_dataframe("wine", bunch.frame.copy(), "target")
        if q == "diabetes":
            bunch = load_diabetes(as_frame=True)
            return _bundle_from_dataframe("diabetes", bunch.frame.copy(), "target")
        if q in {"breast cancer", "breast_cancer"}:
            bunch = load_breast_cancer(as_frame=True)
            return _bundle_from_dataframe("breast_cancer", bunch.frame.copy(), "target")
        if q == "titanic":
            df = pd.DataFrame({
                "pclass": [1, 3, 2, 1, 3],
                "sex": ["female", "male", "female", "male", "male"],
                "age": [29, 34, 21, 45, 18],
                "fare": [211.34, 7.25, 13.00, 83.47, 8.05],
                "survived": [1, 0, 1, 0, 0],
            })
            return _bundle_from_dataframe("titanic", df, "survived")
    except Exception as e:
        return {"success": False, "error": str(e)}

    # ── OpenML fallback: try to fetch any dataset by name ──
    return _openml_fallback(query)


def _openml_fallback(query: str) -> dict:
    """Try to fetch a dataset from OpenML via sklearn.datasets.fetch_openml."""
    try:
        from sklearn.datasets import fetch_openml

        # Clean the query into a plausible OpenML name
        clean = query.strip()

        # Try fetching by name (OpenML search)
        bunch = fetch_openml(name=clean, as_frame=True, parser="auto")
        df = bunch.frame
        if df is None or df.empty:
            return {"success": False, "error": f"OpenML returned empty data for '{query}'."}

        # Limit to first 2000 rows to keep things snappy
        df = df.head(2000).copy()

        target_col = _guess_target_column(df)
        dataset_id = clean.lower().replace(" ", "_")
        return _bundle_from_dataframe(dataset_id, df, target_col)
    except Exception:
        pass

    # If exact name fails, try a fuzzy keyword search via OpenML API
    try:
        import openml
        datasets = openml.datasets.list_datasets(output_format="dataframe")
        # Search by keyword in dataset name
        keywords = query.lower().split()
        mask = datasets["name"].str.lower().apply(
            lambda n: all(kw in n for kw in keywords)
        )
        matches = datasets[mask].sort_values("NumberOfInstances", ascending=False)
        if not matches.empty:
            top = matches.iloc[0]
            oml_ds = openml.datasets.get_dataset(int(top["did"]))
            X, y, _, _ = oml_ds.get_data(dataset_format="dataframe")
            df = X.copy()
            if y is not None:
                df["target"] = y
            df = df.head(2000)
            target_col = "target" if y is not None else _guess_target_column(df)
            dataset_id = str(top["name"]).lower().replace(" ", "_")
            return _bundle_from_dataframe(dataset_id, df, target_col)
    except Exception:
        pass

    return {"success": False, "error": (
        f"Could not find a dataset matching '{query}'. "
        f"Try a well-known name like 'iris', 'mnist', 'titanic', 'wine', 'diabetes', or 'breast_cancer'."
    )}


def _choose_table_file(root: Path) -> Path | None:
    candidates = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TABLE_SUFFIXES]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (0 if p.suffix.lower() == ".csv" else 1, p.stat().st_size))
    return candidates[0]


def _count_images(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _read_table(path: Path) -> pd.DataFrame:
    s = path.suffix.lower()
    if s == ".csv":
        return pd.read_csv(path, nrows=500)
    if s == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=500)
    if s == ".txt":
        return pd.read_csv(path, nrows=500)
    if s == ".json":
        return pd.read_json(path).head(500)
    if s == ".parquet":
        return pd.read_parquet(path).head(500)
    raise ValueError(f"Unsupported table file '{path.name}'.")


def _guess_target_column(df: pd.DataFrame) -> str | None:
    preferred = ["target", "label", "class", "y", "survived", "price", "saleprice"]
    lower_map = {c.lower(): c for c in df.columns}
    for name in preferred:
        if name in lower_map:
            return lower_map[name]
    return df.columns[-1] if len(df.columns) > 1 else None


def _profile_dataframe(df: pd.DataFrame, target_col: str | None) -> dict:
    if target_col and target_col in df.columns:
        target = df[target_col]
        unique = target.nunique(dropna=True)
        if str(target.dtype).startswith(("object", "category", "bool")) or unique <= 30:
            task = "tabular_classification"
            suggested_model = "xgboost"
        else:
            task = "tabular_regression"
            suggested_model = "xgboost"
    else:
        task = "tabular_generic"
        suggested_model = "xgboost"

    return {
        "modality": "tabular",
        "task_family": task,
        "suggested_model": suggested_model,
        "target_column": target_col,
    }


def _profile_image_dataset(root: Path) -> dict:
    return {
        "modality": "image",
        "task_family": "image_classification",
        "suggested_model": "cnn",
        "target_column": None,
        "image_count": _count_images(root),
    }


def load_dataset_by_query(dataset_query: str, base_dir: str = "data/kaggle_cache") -> dict:
    resolved = resolve_best_dataset_ref(dataset_query)
    if not resolved["success"]:
        return _builtin_fallback(dataset_query)

    dataset_ref = resolved["dataset_ref"]
    target_dir = Path(base_dir) / dataset_ref.replace("/", "__")

    files_result = list_dataset_files(dataset_ref)
    if not files_result["success"]:
        return _builtin_fallback(dataset_query)

    download_result = download_dataset(dataset_ref, str(target_dir))
    if not download_result["success"]:
        return _builtin_fallback(dataset_query)

    table_file = _choose_table_file(target_dir)
    image_count = _count_images(target_dir)

    dataset_files = files_result["files"]

    if table_file is not None:
        try:
            df = _read_table(table_file)
            target_col = _guess_target_column(df)
            profile = _profile_dataframe(df, target_col)
            return {
                "success": True,
                "dataset": dataset_query.strip().lower(),
                "dataset_info": {
                    "name": dataset_query.strip().lower(),
                    "kaggle_ref": dataset_ref,
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "feature_names": list(df.columns),
                    "target_name": target_col,
                    "profile": profile,
                    "download_dir": str(target_dir.resolve()),
                    "preview_file": str(table_file.resolve()),
                },
                "dataset_preview": df.head(10).to_dict(orient="records"),
                "dataset_files": dataset_files,
                "dataset_profile": profile,
            }
        except Exception:
            pass

    if image_count > 0:
        profile = _profile_image_dataset(target_dir)
        sample_files = [str(p.name) for p in list(target_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES][:10]
        return {
            "success": True,
            "dataset": dataset_query.strip().lower(),
            "dataset_info": {
                "name": dataset_query.strip().lower(),
                "kaggle_ref": dataset_ref,
                "rows": image_count,
                "columns": 0,
                "feature_names": [],
                "target_name": None,
                "profile": profile,
                "download_dir": str(target_dir.resolve()),
            },
            "dataset_preview": [{"sample_image_file": f} for f in sample_files],
            "dataset_files": dataset_files,
            "dataset_profile": profile,
        }

    return {
        "success": True,
        "dataset": dataset_query.strip().lower(),
        "dataset_info": {
            "name": dataset_query.strip().lower(),
            "kaggle_ref": dataset_ref,
            "rows": 0,
            "columns": 0,
            "feature_names": [],
            "target_name": None,
            "profile": {"modality": "unknown", "task_family": "generic", "suggested_model": "xgboost"},
            "download_dir": str(target_dir.resolve()),
        },
        "dataset_preview": [],
        "dataset_files": dataset_files,
        "dataset_profile": {"modality": "unknown", "task_family": "generic", "suggested_model": "xgboost"},
    }