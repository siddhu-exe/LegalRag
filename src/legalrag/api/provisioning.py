"""
Startup provisioning of production retrieval artifacts from Hugging Face Hub.

Production containers start without the large frozen artifacts. At startup the service
downloads any missing artifacts (bm25.pkl, dense.index, legal_chunks.parquet) using the
existing ``scripts/download_artifacts.py`` helper, which already validates that every
required artifact is present and non-empty. Existing artifacts are reused untouched and
are never baked into the Docker image or committed to Git.

If provisioning fails, the caller must treat the service as not ready (and must never
serve stub responses). Secrets (HF tokens) are never logged or placed on the command line.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from legalrag.api.config import Settings

logger = logging.getLogger(__name__)

REQUIRED_ARTIFACT_NAMES: Sequence[str] = (
    "bm25.pkl",
    "dense.index",
    "legal_chunks.parquet",
)


class ArtifactProvisioningError(RuntimeError):
    """Raised when required production artifacts cannot be provisioned or verified."""


def _repo_root() -> Path:
    """Best-effort resolution of the repository root containing the scripts/ directory."""
    return Path(__file__).resolve().parents[3]


def download_script_path() -> Path:
    """Resolves the existing artifact download helper script."""
    candidates = [
        _repo_root() / "scripts" / "download_artifacts.py",
        Path.cwd() / "scripts" / "download_artifacts.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def missing_artifacts(artifact_dir: Path) -> List[str]:
    """Returns required artifact filenames that are absent or empty in the target directory."""
    missing: List[str] = []
    for filename in REQUIRED_ARTIFACT_NAMES:
        path = artifact_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(filename)
    return missing


def _scrub_secrets(text: str, secrets: Sequence[Optional[str]]) -> str:
    """Removes known secret values from a string before it is logged."""
    scrubbed = text
    for secret in secrets:
        if secret and len(secret) >= 8:
            scrubbed = scrubbed.replace(secret, "***")
    return scrubbed


def _default_runner(cmd: Sequence[str], env: dict, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(_repo_root()),
    )


def provision_production_artifacts(
    settings: Settings,
    runner: Optional[Callable[[Sequence[str], dict, int], subprocess.CompletedProcess]] = None,
) -> None:
    """
    Ensures the required production artifacts exist locally, downloading them from
    Hugging Face Hub when missing.

    Raises:
        ArtifactProvisioningError: if artifacts are missing and cannot be provisioned or
            verified. The caller must not initialize/serve the production pipeline.
    """
    artifact_dir = Path(settings.artifact_dir)
    missing = missing_artifacts(artifact_dir)
    if not missing:
        logger.info("Production artifacts already present in '%s'; reusing them.", artifact_dir)
        return

    logger.info(
        "Missing production artifacts (%s); provisioning from Hugging Face Hub.",
        ", ".join(missing),
    )

    if not settings.hf_repo_id or not settings.hf_repo_id.strip():
        raise ArtifactProvisioningError(
            "Production artifacts are missing and HF_REPO_ID is not configured."
        )

    script = download_script_path()
    if not script.is_file():
        raise ArtifactProvisioningError("Artifact download script is not available in the image.")

    cmd = [
        sys.executable,
        str(script),
        "--repo-id",
        settings.hf_repo_id,
        "--target-dir",
        str(artifact_dir),
    ]

    env = dict(os.environ)
    if settings.hf_token:
        # Passed via environment only; never logged and never placed in argv.
        env["HF_TOKEN"] = settings.hf_token

    run = runner or _default_runner
    try:
        result = run(cmd, env, settings.artifact_download_timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise ArtifactProvisioningError("Timed out while downloading production artifacts.") from exc
    except Exception as exc:
        raise ArtifactProvisioningError("Failed to run production artifact download.") from exc

    if getattr(result, "returncode", 1) != 0:
        output = _scrub_secrets(
            f"{getattr(result, 'stdout', '') or ''}\n{getattr(result, 'stderr', '') or ''}",
            [settings.hf_token],
        ).strip()
        if output:
            logger.error("Artifact provisioning output:\n%s", output[-2000:])
        raise ArtifactProvisioningError("Production artifact download failed.")

    still_missing = missing_artifacts(artifact_dir)
    if still_missing:
        raise ArtifactProvisioningError(
            "Production artifact verification failed for: " + ", ".join(still_missing)
        )

    logger.info("Production artifacts provisioned successfully in '%s'.", artifact_dir)
