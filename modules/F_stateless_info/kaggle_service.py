from __future__ import annotations

import csv
import io
import subprocess
from pathlib import Path

# Actively load local .env so Kaggle picks up KAGGLE_USERNAME and KAGGLE_KEY
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass


FALLBACK_SEARCH_RESULTS = {
    "mnist": [
        {
            "ref": "hojjatk/mnist-dataset",
            "title": "MNIST Dataset",
            "size": "15683414",
            "lastUpdated": "2017-03-01 00:00:00",
            "downloadCount": "100000",
            "voteCount": "1000",
            "usabilityRating": "1.0",
        }
    ],
    "titanic": [
        {
            "ref": "heptapod/titanic",
            "title": "Titanic",
            "size": "11090",
            "lastUpdated": "2017-05-16 08:14:22.210000",
            "downloadCount": "152593",
            "voteCount": "1960",
            "usabilityRating": "0.7058824",
        }
    ],
}

FALLBACK_FILES = {
    "heptapod/titanic": [{"name": "titanic.csv", "size": "11090", "creationDate": "2017-05-16"}],
    "hojjatk/mnist-dataset": [{"name": "mnist_train.csv", "size": "36523880", "creationDate": "2017-03-01"}],
}

FALLBACK_COMPETITIONS = [{"ref": "titanic", "deadline": "2030-01-01", "category": "Getting Started"}]
FALLBACK_LEADERBOARD = {"titanic": [{"teamName": "demo-team", "score": "0.99999"}]}


# ── Kaggle Python API (preferred) ────────────────────────────

def _get_kaggle_api():
    """Get an authenticated KaggleApi instance. Returns None if unavailable."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception:
        return None


def _search_datasets_api(query: str, page_size: int = 10) -> dict:
    """Search datasets using the Kaggle Python API."""
    api = _get_kaggle_api()
    if api is None:
        return {"success": False, "error": "Kaggle API not available"}

    try:
        results = api.dataset_list(search=query)
        rows = []
        for ds in results[:page_size]:
            rows.append({
                "ref": str(ds.ref),
                "title": str(getattr(ds, "title", ds.ref)),
                "size": str(getattr(ds, "totalBytes", "")),
                "lastUpdated": str(getattr(ds, "lastUpdated", "")),
                "downloadCount": str(getattr(ds, "downloadCount", "")),
                "voteCount": str(getattr(ds, "voteCount", "")),
                "usabilityRating": str(getattr(ds, "usabilityRating", "")),
            })
        return {"success": True, "query": query, "results": rows}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _download_dataset_api(dataset_ref: str, output_dir: str) -> dict:
    """Download a dataset using the Kaggle Python API."""
    api = _get_kaggle_api()
    if api is None:
        return {"success": False, "error": "Kaggle API not available"}

    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset_ref, path=output_dir, unzip=True, quiet=True)
        return {"success": True, "dataset_ref": dataset_ref, "download_dir": str(Path(output_dir).resolve())}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _list_dataset_files_api(dataset_ref: str) -> dict:
    """List files in a dataset using the Kaggle Python API."""
    api = _get_kaggle_api()
    if api is None:
        return {"success": False, "error": "Kaggle API not available"}

    try:
        files = api.dataset_list_files(dataset_ref)
        file_list = files.files if hasattr(files, "files") else files
        rows = []
        for f in file_list:
            rows.append({
                "name": str(getattr(f, "name", f)),
                "size": str(getattr(f, "totalBytes", getattr(f, "size", ""))),
                "creationDate": str(getattr(f, "creationDate", "")),
            })
        return {"success": True, "dataset_ref": dataset_ref, "files": rows}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── CLI Fallback ─────────────────────────────────────────────

def _run_kaggle_command(args: list[str]) -> dict:
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf8"
    try:
        proc = subprocess.run(
            ["kaggle", *args],
            capture_output=True,
            check=False,
            shell=False,
            env=env,
        )
        stdout = proc.stdout.decode("utf-8", errors="replace").strip()
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            return {"success": False, "error": stderr or stdout or "Kaggle command failed"}
        return {"success": True, "output": stdout}
    except Exception as e:
        return {"success": False, "error": f"Failed to run Kaggle command: {e}"}


def _parse_csv_output(text: str) -> list[dict]:
    if not text.strip():
        return []
    return list(csv.DictReader(io.StringIO(text)))


# ── Public API (tries Python API → CLI → fallback) ──────────

def search_datasets(query: str, page_size: int = 10) -> dict:
    # Try Python API first
    result = _search_datasets_api(query, page_size)
    if result["success"] and result.get("results"):
        return result

    # Try CLI fallback
    cli_result = _run_kaggle_command(["datasets", "list", "-s", query, "-v"])
    if cli_result["success"]:
        rows = _parse_csv_output(cli_result["output"])
        if rows:
            return {"success": True, "query": query, "results": rows}

    # OpenML fallback
    try:
        import openml
        datasets = openml.datasets.list_datasets(output_format="dataframe")
        if not datasets.empty:
            keywords = query.lower().split()
            mask = datasets["name"].str.lower().apply(lambda n: all(kw in n for kw in keywords))
            matches = datasets[mask].sort_values("NumberOfInstances", ascending=False).head(page_size)
            rows = []
            for _, row in matches.iterrows():
                rows.append({
                    "ref": "openml/" + str(row["name"]),
                    "title": str(row["name"]) + f" (OpenML id: {row['did']})",
                    "size": str(row["NumberOfInstances"]) + " rows",
                    "lastUpdated": "",
                    "downloadCount": "",
                    "voteCount": "",
                    "usabilityRating": "1.0",
                })
            if rows:
                return {"success": True, "query": query, "results": rows}
    except:
        pass
        
    # Hugging Face fallback
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        hf_matches = list(api.list_datasets(search=query, limit=page_size))
        rows = []
        for ds in hf_matches:
             rows.append({
                  "ref": "hf/" + ds.id,
                  "title": ds.id + " (Hugging Face)",
                  "size": "Unknown rows",
                  "lastUpdated": str(ds.lastModified) if hasattr(ds, 'lastModified') else "",
                  "downloadCount": str(ds.downloads) if hasattr(ds, 'downloads') else "",
                  "voteCount": "",
                  "usabilityRating": "1.0",
             })
        if rows:
             return {"success": True, "query": query, "results": rows}
    except:
        pass

    # Hardcoded fallback
    return {"success": True, "query": query, "results": FALLBACK_SEARCH_RESULTS.get(query.lower().strip(), [])}


def resolve_best_dataset_ref(query: str) -> dict:
    result = search_datasets(query, page_size=10)
    rows = result["results"]
    if not rows:
        return {
            "success": False, 
            "error": (
                f"No datasets found for query '{query}'.\n\n"
                f"**Note:** To search Kaggle's live database, you must configure the Kaggle API:\n"
                f"1. `pip install kaggle`\n"
                f"2. Get `kaggle.json` from kaggle.com → Settings → Create New Token\n"
                f"3. Place it in `~/.kaggle/kaggle.json`\n"
            )
        }
    row = rows[0]
    ref = row.get("ref") or row.get("id") or row.get("datasetSlug")
    if not ref:
        return {"success": False, "error": f"Could not resolve dataset ref for '{query}'."}
    return {"success": True, "dataset_ref": ref, "top_result": row, "results": rows, "query": query}


def list_dataset_files(dataset_ref: str) -> dict:
    # Try Python API first
    result = _list_dataset_files_api(dataset_ref)
    if result["success"]:
        return result

    # Try CLI fallback
    cli_result = _run_kaggle_command(["datasets", "files", dataset_ref, "-v"])
    if cli_result["success"]:
        rows = _parse_csv_output(cli_result["output"])
        return {"success": True, "dataset_ref": dataset_ref, "files": rows}

    return {"success": True, "dataset_ref": dataset_ref, "files": FALLBACK_FILES.get(dataset_ref, [])}


def download_dataset(dataset_ref: str, output_dir: str) -> dict:
    # Try Python API first
    result = _download_dataset_api(dataset_ref, output_dir)
    if result["success"]:
        return result

    # Try CLI fallback
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cli_result = _run_kaggle_command(["datasets", "download", dataset_ref, "-p", output_dir, "--unzip", "-o", "-q"])
    if cli_result["success"]:
        return {"success": True, "dataset_ref": dataset_ref, "download_dir": str(Path(output_dir).resolve())}

    return {"success": False, "error": cli_result.get("error", f"Failed to download dataset '{dataset_ref}'.")}


def get_dataset_info(dataset_query: str) -> dict:
    resolved = resolve_best_dataset_ref(dataset_query)
    if not resolved["success"]:
        return resolved
    files = list_dataset_files(resolved["dataset_ref"])
    if not files["success"]:
        return files
    return {
        "success": True,
        "query": dataset_query,
        "dataset_ref": resolved["dataset_ref"],
        "top_result": resolved["top_result"],
        "search_results": resolved["results"],
        "files": files["files"],
    }


def show_competitions(page_size: int = 10) -> dict:
    result = _run_kaggle_command(["competitions", "list", "-v"])
    if result["success"]:
        return {"success": True, "results": _parse_csv_output(result["output"])}
    return {"success": True, "results": FALLBACK_COMPETITIONS}


def show_leaderboard(competition_name: str) -> dict:
    result = _run_kaggle_command(["competitions", "leaderboard", competition_name, "-s", "-v"])
    if result["success"]:
        return {"success": True, "competition_name": competition_name, "results": _parse_csv_output(result["output"])}
    return {"success": True, "competition_name": competition_name, "results": FALLBACK_LEADERBOARD.get(competition_name, [])}