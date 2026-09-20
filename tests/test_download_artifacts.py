"""
Regression tests for the Hugging Face artifact downloader.

The production artifacts live in a Hugging Face DATASET repository
(``siddhu23/LegalRag_Dataset``), so ``hf_hub_download`` must be called with
``repo_type="dataset"``. Without it, Hugging Face queries the model-repository API
and reports every artifact as missing, which left Azure /ready permanently not ready.

No network calls and no downloads are performed; ``hf_hub_download`` is mocked.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_download_artifacts_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "download_artifacts.py"
    spec = importlib.util.spec_from_file_location("download_artifacts_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


download_artifacts_module = _load_download_artifacts_module()


def _fake_hf_hub_download(**kwargs):
    """Writes a small placeholder file and returns its path, mimicking the real API."""
    destination = Path(kwargs["local_dir"]) / kwargs["filename"]
    destination.write_bytes(b"placeholder")
    return str(destination)


class TestDownloadArtifactsRepoType(unittest.TestCase):
    def test_hf_hub_download_uses_dataset_repo_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            mocked = MagicMock(side_effect=_fake_hf_hub_download)
            with patch.object(download_artifacts_module, "hf_hub_download", mocked):
                success = download_artifacts_module.download_artifacts(
                    repo_id="siddhu23/LegalRag_Dataset",
                    target_dir=Path(tmp),
                    token="hf_test_token_123456",
                )

            self.assertTrue(success)
            self.assertEqual(mocked.call_count, len(download_artifacts_module.REQUIRED_ARTIFACTS))
            self.assertEqual(
                {call.kwargs["filename"] for call in mocked.call_args_list},
                set(download_artifacts_module.REQUIRED_ARTIFACTS),
            )
            for call in mocked.call_args_list:
                self.assertEqual(call.kwargs["repo_type"], "dataset")
                self.assertEqual(call.kwargs["repo_id"], "siddhu23/LegalRag_Dataset")

    def test_repo_type_is_dataset_for_every_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            mocked = MagicMock(side_effect=_fake_hf_hub_download)
            with patch.object(download_artifacts_module, "hf_hub_download", mocked):
                download_artifacts_module.download_artifacts(
                    repo_id="siddhu23/LegalRag_Dataset",
                    target_dir=Path(tmp),
                    token="hf_test_token_123456",
                )

            repo_types = [call.kwargs.get("repo_type") for call in mocked.call_args_list]
            self.assertTrue(repo_types)
            self.assertTrue(all(repo_type == "dataset" for repo_type in repo_types))

    def test_environment_token_is_used_and_never_logged(self):
        secret = "hf_super_secret_token_9876543210"
        with tempfile.TemporaryDirectory() as tmp:
            mocked = MagicMock(side_effect=_fake_hf_hub_download)
            with patch.object(download_artifacts_module, "hf_hub_download", mocked), patch.dict(
                "os.environ", {"HF_TOKEN": secret}, clear=False
            ), self.assertLogs(download_artifacts_module.logger, level="INFO") as captured:
                success = download_artifacts_module.download_artifacts(
                    repo_id="siddhu23/LegalRag_Dataset",
                    target_dir=Path(tmp),
                    token=None,
                )

            self.assertTrue(success)
            for call in mocked.call_args_list:
                self.assertEqual(call.kwargs["token"], secret)
            self.assertNotIn(secret, "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
