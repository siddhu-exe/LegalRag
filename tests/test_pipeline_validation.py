"""Retrieval-only validation: the generator requirement is optional in the shared code (review fix 1)."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from legalrag.api.config import Settings
from legalrag.api.dependencies import (
    PipelineComponents,
    StubBM25Retriever,
    StubCrossEncoderReranker,
    StubDenseRetriever,
    create_stub_chunks,
    validate_pipeline_components,
)


def production_pipeline(generator=None, chunk_ids=None):
    chunks = create_stub_chunks()
    ids = chunk_ids if chunk_ids is not None else chunks["chunk_id"].tolist()
    return PipelineComponents(
        bm25=StubBM25Retriever(chunk_ids=ids),
        dense=StubDenseRetriever(chunk_ids=ids),
        reranker=StubCrossEncoderReranker(),
        chunks=chunks,
        generator=generator,
        environment="production",
    )


class TestGeneratorRequirement(unittest.TestCase):
    def test_default_still_requires_a_generator_so_serving_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "no generation client"):
            validate_pipeline_components(production_pipeline(generator=None))

    def test_retrieval_only_validation_accepts_a_missing_generator(self):
        validate_pipeline_components(production_pipeline(generator=None), require_generator=False)

    def test_a_present_generator_is_accepted_either_way(self):
        validate_pipeline_components(production_pipeline(generator=object()))
        validate_pipeline_components(production_pipeline(generator=object()), require_generator=False)

    def test_optional_generator_does_not_relax_the_chunk_id_check(self):
        broken = production_pipeline(generator=None, chunk_ids=["ghost_chunk"])
        with self.assertRaisesRegex(ValueError, "absent from"):
            validate_pipeline_components(broken, require_generator=False)


class TestRetrievalEvalProductionLoader(unittest.TestCase):
    """load_pipeline('production') used to fail on every real run because it has no generator."""

    def test_production_load_succeeds_without_a_generator_or_groq_key(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "scripts" / "run_retrieval_eval.py"
        spec = importlib.util.spec_from_file_location("run_retrieval_eval_loader_test", path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)

        chunks = create_stub_chunks()
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("bm25.pkl", "dense.index", "legal_chunks.parquet"):
                (Path(tmp) / name).write_bytes(b"x")  # only existence is checked; loaders are patched
            settings = Settings(_env_file=None, environment="production", artifact_dir=Path(tmp), groq_api_key=None)
            bm25 = StubBM25Retriever(chunk_ids=chunks["chunk_id"].tolist())
            dense = StubDenseRetriever(chunk_ids=chunks["chunk_id"].tolist())
            with mock.patch("legalrag.api.config.get_settings", return_value=settings), \
                    mock.patch("pandas.read_parquet", return_value=chunks), \
                    mock.patch("legalrag.retrieval.bm25.BM25Retriever.load", return_value=bm25), \
                    mock.patch("legalrag.retrieval.dense.DenseRetriever.load_index", return_value=dense):
                pipeline, lookup = script.load_pipeline("production")
        self.assertIsNone(pipeline.generator)
        self.assertEqual(pipeline.environment, "production")
        self.assertIn("stub_chunk_0", lookup.get_many(["stub_chunk_0"]))

    def test_production_load_still_rejects_ids_that_do_not_resolve(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "scripts" / "run_retrieval_eval.py"
        spec = importlib.util.spec_from_file_location("run_retrieval_eval_loader_test2", path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)

        chunks = create_stub_chunks()
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("bm25.pkl", "dense.index", "legal_chunks.parquet"):
                (Path(tmp) / name).write_bytes(b"x")
            settings = Settings(_env_file=None, environment="production", artifact_dir=Path(tmp))
            ghost = StubBM25Retriever(chunk_ids=["ghost"])
            dense = StubDenseRetriever(chunk_ids=chunks["chunk_id"].tolist())
            with mock.patch("legalrag.api.config.get_settings", return_value=settings), \
                    mock.patch("pandas.read_parquet", return_value=chunks), \
                    mock.patch("legalrag.retrieval.bm25.BM25Retriever.load", return_value=ghost), \
                    mock.patch("legalrag.retrieval.dense.DenseRetriever.load_index", return_value=dense):
                with self.assertRaisesRegex(ValueError, "absent from"):
                    script.load_pipeline("production")


if __name__ == "__main__":
    unittest.main()
