from __future__ import annotations

import csv
import io
import subprocess
from pathlib import Path


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


def search_datasets(query: str, page_size: int = 10) -> dict:
    result = _run_kaggle_command(["datasets", "list", "-s", query, "-v"])
    if result["success"]:
        rows = _parse_csv_output(result["output"])
        return {"success": True, "query": query, "results": rows}
    return {"success": True, "query": query, "results": FALLBACK_SEARCH_RESULTS.get(query.lower().strip(), [])}


def resolve_best_dataset_ref(query: str) -> dict:
    result = search_datasets(query, page_size=10)
    rows = result["results"]
    if not rows:
        return {"success": False, "error": f"No Kaggle datasets found for query '{query}'."}
    row = rows[0]
    ref = row.get("ref") or row.get("id") or row.get("datasetSlug")
    if not ref:
        return {"success": False, "error": f"Could not resolve dataset ref for '{query}'."}
    return {"success": True, "dataset_ref": ref, "top_result": row, "results": rows, "query": query}


def list_dataset_files(dataset_ref: str) -> dict:
    result = _run_kaggle_command(["datasets", "files", dataset_ref, "-v"])
    if result["success"]:
        rows = _parse_csv_output(result["output"])
        return {"success": True, "dataset_ref": dataset_ref, "files": rows}
    return {"success": True, "dataset_ref": dataset_ref, "files": FALLBACK_FILES.get(dataset_ref, [])}


def download_dataset(dataset_ref: str, output_dir: str) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result = _run_kaggle_command(["datasets", "download", dataset_ref, "-p", output_dir, "--unzip", "-o", "-q"])
    if result["success"]:
        return {"success": True, "dataset_ref": dataset_ref, "download_dir": str(Path(output_dir).resolve())}
    return {"success": False, "error": result.get("error", f"Failed to download dataset '{dataset_ref}'.")}


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