#!/usr/bin/env python3
"""
Hugging Face Hub artifact downloader for production LegalRAG deployment.

Downloads the locked, frozen production artifacts required for full retrieval:
- 'bm25.pkl': Lexical BM25 index (~579 MB)
- 'dense.index': FAISS FlatIP 768-dim dense vector index (~1.65 GB)
- 'legal_chunks.parquet': Chunk metadata and full text (~261 MB)

Usage:
    python scripts/download_artifacts.py --repo-id <hf-username>/<repo-name>
    python scripts/download_artifacts.py --repo-id <hf-username>/<repo-name> --target-dir ./artifacts
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("download_artifacts")

REQUIRED_ARTIFACTS = [
    "bm25.pkl",
    "dense.index",
    "legal_chunks.parquet",
]


def download_artifacts(
    repo_id: str,
    target_dir: Path,
    token: Optional[str] = None,
    filenames: Optional[List[str]] = None,
) -> bool:
    """
    Downloads required artifacts from Hugging Face Hub to the local target directory.
    """
    if hf_hub_download is None:
        logger.error(
            "huggingface_hub package is not installed. Install via: pip install huggingface_hub"
        )
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    artifacts_to_fetch = filenames or REQUIRED_ARTIFACTS
    resolved_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    logger.info("Downloading %d production artifacts from Hub repo: '%s'", len(artifacts_to_fetch), repo_id)
    logger.info("Destination directory: %s", target_dir.resolve())

    all_succeeded = True

    for filename in artifacts_to_fetch:
        dest_path = target_dir / filename
        if dest_path.is_file() and dest_path.stat().st_size > 0:
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            logger.info("Artifact already exists locally: %s (%.2f MB), skipping download.", filename, size_mb)
            continue

        logger.info("Fetching '%s' from repository...", filename)
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target_dir),
                token=resolved_token,
            )
            file_size_mb = Path(downloaded_path).stat().st_size / (1024 * 1024)
            logger.info("Successfully fetched '%s' (%.2f MB)", filename, file_size_mb)
        except Exception as exc:
            logger.error("Failed to download '%s': %s", filename, str(exc))
            all_succeeded = False

    return all_succeeded


def verify_artifacts(target_dir: Path) -> bool:
    """
    Verifies that all required artifacts exist in the target directory and are non-empty.
    """
    missing = []
    for filename in REQUIRED_ARTIFACTS:
        file_path = target_dir / filename
        if not file_path.is_file():
            missing.append(f"{filename} (not found)")
        elif file_path.stat().st_size == 0:
            missing.append(f"{filename} (0 bytes / empty)")

    if missing:
        logger.error("Artifact verification failed. Missing files:\n" + "\n".join(f"  - {m}" for m in missing))
        return False

    logger.info("All required production artifacts verified successfully in %s", target_dir.resolve())
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download LegalRAG production retrieval artifacts from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=os.getenv("HF_REPO_ID"),
        help="Hugging Face Hub repository ID (e.g. 'username/legalrag-artifacts').",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=os.getenv("ARTIFACT_DIR", "artifacts"),
        help="Local directory where artifacts should be saved (default: 'artifacts').",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face Hub token for private repositories or rate limits.",
    )

    args = parser.parse_args()

    if not args.repo_id:
        logger.error(
            "Hugging Face repository ID is required. Pass --repo-id or set the HF_REPO_ID environment variable."
        )
        sys.exit(1)

    target_dir = Path(args.target_dir)
    success = download_artifacts(
        repo_id=args.repo_id,
        target_dir=target_dir,
        token=args.token,
    )

    if not success:
        logger.error("One or more artifact downloads failed.")
        sys.exit(1)

    if not verify_artifacts(target_dir):
        sys.exit(1)


if __name__ == "__main__":
    main()
