"""
The low-memory backends wired into the pipeline loader, the query endpoint and the retrieval evaluation
(plan task C4): artifact contract tests for the v2 layout.

Real artifacts are built in a temporary folder from the six-chunk test corpus: a bm25s index, a SQLite
chunk store, a FAISS index (flat and IVF) and, where PyTorch is installed, real ONNX models exported
from a tiny randomly initialised BERT. No model is downloaded and no real data is read. Every flag
defaults to the v1 behaviour, and a test pins that.
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_artifact_contracts import (
    CORPUS,
    ContextEchoGenerator,
    FakeEmbedder,
    build_chunks_dataframe,
)
from test_onnx_models import HAVE_EXPORT_STACK, make_tiny_model

from legalrag.api.config import Settings
from legalrag.api.dependencies import (
    PipelineComponents,
    StubCrossEncoderReranker,
    load_production_pipeline,
    load_retrieval_components,
    reset_pipeline_cache,
    validate_pipeline_components,
)
from legalrag.api.routes import query_endpoint
from legalrag.api.schemas import QueryRequest
from legalrag.retrieval.bm25 import BM25Retriever
from legalrag.retrieval.bm25s_backend import Bm25sRetriever, bm25s
from legalrag.retrieval.chunk_ids import ChunkIdArray
from legalrag.retrieval.chunk_store import SqliteChunkStore
from legalrag.retrieval.dense import DenseRetriever
from legalrag.retrieval.faiss_builders import EmbeddingShards, IndexSpec, build_index, write_index
from legalrag.retrieval.onnx_models import OnnxEmbedder, OnnxReranker

CHUNK_IDS = [cid for cid, _ in CORPUS]
TEXTS = [text for _, text in CORPUS]
QUESTION = "Anticipatory bail under Section 438 CrPC apprehension of arrest"
needs_bm25s = unittest.skipIf(bm25s is None, "bm25s is not installed")


def settings_for(artifact_dir: Path, **flags) -> Settings:
    return Settings(_env_file=None, environment="production", groq_api_key="gsk_dummy_key_for_test",
                    artifact_dir=artifact_dir, model_warmup_enabled=False, **flags)


def write_v1_style_dense(artifact_dir: Path, embedder=None, index_kind="flat", nlist=2) -> None:
    """dense.index over the six chunks, embedded with ``embedder`` (default: the vocabulary-overlap fake)."""
    embedder = embedder or FakeEmbedder()
    vectors = embedder.encode(TEXTS, normalize_embeddings=True).astype(np.float32)
    path = artifact_dir / "dense.index"
    if index_kind == "flat":
        retriever = DenseRetriever(model_name="fake")
        retriever.build_index(vectors, CHUNK_IDS)
        retriever.save_index(path)
    else:
        np.save(artifact_dir / "emb.npy", vectors)
        index, meta = build_index(EmbeddingShards([artifact_dir / "emb.npy"]),
                                  IndexSpec("ivf_sq8", vectors.shape[1], nlist), train_size=len(vectors), nprobe=nlist)
        write_index(index, path, meta)


def write_v2_artifacts(artifact_dir: Path, dense="flat", bm25="bm25s", store="sqlite", embedder=None) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if bm25 == "bm25s" and bm25s is not None:
        Bm25sRetriever.build(CHUNK_IDS, TEXTS).save(artifact_dir / "bm25s")
    if store == "sqlite":
        SqliteChunkStore.build(artifact_dir / "chunks" / "chunks.sqlite", build_chunks_dataframe().to_dict("records")).close()
    build_chunks_dataframe().to_parquet(artifact_dir / "legal_chunks.parquet", index=False)
    BM25Retriever(chunk_ids=CHUNK_IDS, corpus_texts=TEXTS).save(artifact_dir / "bm25.pkl")  # so the v1 stack loads too
    write_v1_style_dense(artifact_dir, embedder, dense)


def ask(pipeline: PipelineComponents):
    pipeline.generator = ContextEchoGenerator()
    return query_endpoint(QueryRequest.model_validate({"question": QUESTION}), pipeline)


class TestDefaultsAreTheV1Stack(unittest.TestCase):
    def test_every_flag_defaults_to_the_v1_behaviour(self):
        s = Settings(_env_file=None)
        self.assertEqual(
            (s.bm25_backend, s.chunk_store, s.embedder_backend, s.reranker_backend, s.dense_nprobe, s.dense_mmap, s.bm25s_mmap, s.onnx_threads),
            ("rank_bm25", "parquet", "torch", "torch", None, False, True, None),
        )

    def test_paths_follow_the_artifact_directory(self):
        s = Settings(_env_file=None, artifact_dir=Path("artifacts/v2"))
        self.assertEqual(s.bm25s_dir, Path("artifacts/v2/bm25s"))
        self.assertEqual(s.chunk_store_path, Path("artifacts/v2/chunks/chunks.sqlite"))
        self.assertEqual(s.onnx_embedder_dir, Path("artifacts/v2/onnx/embedder"))
        self.assertEqual(s.onnx_reranker_dir, Path("artifacts/v2/onnx/reranker"))
        self.assertEqual(s.bm25_path, Path("artifacts/v2/bm25.pkl"))

    def test_unknown_flag_values_are_rejected(self):
        for flag, value in (("bm25_backend", "lucene"), ("chunk_store", "redis"), ("embedder_backend", "tflite"), ("dense_nprobe", 0)):
            with self.subTest(flag=flag):
                with self.assertRaises(Exception):
                    Settings(_env_file=None, **{flag: value})

    def test_flags_can_come_from_the_environment(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"BM25_BACKEND": "bm25s", "CHUNK_STORE": "sqlite", "DENSE_NPROBE": "7"}):
            s = Settings(_env_file=None)
        self.assertEqual((s.bm25_backend, s.chunk_store, s.dense_nprobe), ("bm25s", "sqlite", 7))


@needs_bm25s
class TestV2BackendsThroughTheLoaderAndEndpoint(unittest.TestCase):
    def setUp(self):
        reset_pipeline_cache()
        self.addCleanup(reset_pipeline_cache)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "v2"
        write_v2_artifacts(self.dir)

    def load(self, **flags):
        pipeline = load_production_pipeline(settings_for(self.dir, **flags))
        pipeline.dense._model = FakeEmbedder()
        pipeline.reranker = StubCrossEncoderReranker()
        return pipeline

    def test_bm25s_backend_is_loaded_memory_mapped_and_answers(self):
        pipeline = self.load(bm25_backend="bm25s")
        self.assertIsInstance(pipeline.bm25, Bm25sRetriever)
        self.assertIsInstance(pipeline.bm25.chunk_ids, ChunkIdArray)
        self.assertEqual(pipeline.bm25.search(QUESTION, 3)[0][0], "PHHC010000012020_chunk_0")
        validate_pipeline_components(pipeline)

    def test_sqlite_store_replaces_the_in_memory_chunk_table(self):
        pipeline = self.load(chunk_store="sqlite")
        self.assertIsInstance(pipeline.bm25, BM25Retriever)  # only the chunk table changed
        self.assertTrue(pipeline.chunks.empty)
        self.assertIsInstance(pipeline.chunk_store, SqliteChunkStore)
        self.assertIsInstance(pipeline.dense.chunk_ids, ChunkIdArray)
        validate_pipeline_components(pipeline)

    def test_sqlite_store_gives_exactly_the_same_answer_as_the_in_memory_table(self):
        in_memory = ask(self.load(bm25_backend="bm25s"))
        on_disk = ask(self.load(bm25_backend="bm25s", chunk_store="sqlite"))
        self.assertEqual(on_disk.status, "ok")
        self.assertEqual(on_disk.retrieved_chunk_ids, in_memory.retrieved_chunk_ids)
        self.assertEqual([c.model_dump() for c in on_disk.citations], [c.model_dump() for c in in_memory.citations])
        self.assertGreater(len(on_disk.citations), 0)
        self.assertEqual(on_disk.citations[0].cnr, "PHHC010000012020")  # metadata came from the SQLite store

    def test_all_v2_flags_together_agree_with_the_v1_stack_on_the_best_chunk(self):
        # The full ranking can differ: rank_bm25 returns zero-score chunks and bm25s does not, which
        # changes the fused list below the top. The best chunk and a valid grounded answer must not change.
        v1 = ask(self.load())
        v2 = ask(self.load(bm25_backend="bm25s", chunk_store="sqlite", dense_nprobe=2, dense_mmap=True))
        self.assertEqual((v1.status, v2.status), ("ok", "ok"))
        self.assertEqual(v2.retrieved_chunk_ids[0], v1.retrieved_chunk_ids[0])
        self.assertEqual(v2.retrieved_chunk_ids[0], "PHHC010000012020_chunk_0")
        self.assertTrue(all(c in CHUNK_IDS for c in v2.retrieved_chunk_ids))
        self.assertTrue(all(c.chunk_id in v2.retrieved_chunk_ids for c in v2.citations))

    def test_approximate_dense_index_is_loaded_with_nprobe_and_works_in_the_cascade(self):
        import faiss

        write_v1_style_dense(self.dir, index_kind="ivf", nlist=2)
        pipeline = self.load(bm25_backend="bm25s", chunk_store="sqlite", dense_nprobe=2, dense_mmap=True)
        self.assertEqual(faiss.extract_index_ivf(pipeline.dense.index).nprobe, 2)
        self.assertEqual(ask(pipeline).retrieved_chunk_ids[0], "PHHC010000012020_chunk_0")

    def test_retrieval_components_have_no_generator(self):
        components = load_retrieval_components(settings_for(self.dir, bm25_backend="bm25s", chunk_store="sqlite"))
        self.assertIsNone(components.generator)
        validate_pipeline_components(components, require_generator=False)

    def test_missing_artifacts_are_listed_for_the_configured_backends(self):
        import shutil

        shutil.rmtree(self.dir / "bm25s")
        (self.dir / "chunks" / "chunk_ids.npy").unlink()
        with self.assertRaises(FileNotFoundError) as ctx:
            load_production_pipeline(settings_for(self.dir, bm25_backend="bm25s", chunk_store="sqlite"))
        message = str(ctx.exception)
        self.assertIn("bm25s index missing", message)
        self.assertIn("Chunk id array missing", message)
        self.assertNotIn("legal_chunks.parquet", message)  # that artifact is not required by this configuration
        (self.dir / "bm25.pkl").unlink()
        with self.assertRaises(FileNotFoundError) as ctx:  # and the v1 configuration still asks for the v1 files
            load_production_pipeline(settings_for(self.dir))
        self.assertIn("BM25 index artifact missing", str(ctx.exception))

    def test_index_and_chunk_table_built_from_different_corpora_fail_closed(self):
        reordered = list(reversed(CHUNK_IDS))
        Bm25sRetriever.build(reordered, list(reversed(TEXTS))).save(self.dir / "bm25s")  # same ids, different order
        with self.assertRaisesRegex(ValueError, "disagree"):
            load_production_pipeline(settings_for(self.dir, bm25_backend="bm25s", chunk_store="sqlite"))
        Bm25sRetriever.build(CHUNK_IDS[:4], TEXTS[:4]).save(self.dir / "bm25s")  # fewer chunks
        with self.assertRaisesRegex(ValueError, "different corpora"):
            load_production_pipeline(settings_for(self.dir, bm25_backend="bm25s", chunk_store="sqlite"))

    def test_a_chunk_store_missing_ids_the_indexes_return_fails_validation(self):
        # bm25s and chunk_ids.npy agree with each other (both renamed one id), but the SQLite rows still
        # carry the old id, so a retriever could return an id the store cannot resolve.
        renamed = list(CHUNK_IDS)
        renamed[3] = "renamed-chunk"
        Bm25sRetriever.build(renamed, TEXTS).save(self.dir / "bm25s")
        ChunkIdArray.from_ids(renamed).save(self.dir / "chunks" / "chunk_ids.npy")
        components = load_retrieval_components(settings_for(self.dir, chunk_store="sqlite", bm25_backend="bm25s"))
        with self.assertRaisesRegex(ValueError, "absent from the chunk store"):
            validate_pipeline_components(components, require_generator=False)

    def test_store_size_must_match_the_indexes(self):
        SqliteChunkStore.build(self.dir / "chunks" / "chunks.sqlite", build_chunks_dataframe().to_dict("records")[:4]).close()
        # The chunk id array now lists 4 ids but dense.index holds 6 vectors.
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            load_production_pipeline(settings_for(self.dir, chunk_store="sqlite"))
        # With bm25s as well, the disagreement is caught even earlier, between the two id lists.
        with self.assertRaisesRegex(ValueError, "different corpora"):
            load_production_pipeline(settings_for(self.dir, chunk_store="sqlite", bm25_backend="bm25s"))


@needs_bm25s
class TestRetrievalEvalUsesTheSameBackends(unittest.TestCase):
    def test_load_pipeline_follows_the_flags_and_scores_with_the_store(self):
        from unittest import mock

        path = Path(__file__).resolve().parents[1] / "scripts" / "run_retrieval_eval.py"
        spec = importlib.util.spec_from_file_location("run_retrieval_eval_wiring", path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "v2"
            write_v2_artifacts(directory)
            settings = settings_for(directory, bm25_backend="bm25s", chunk_store="sqlite")
            with mock.patch("legalrag.api.config.get_settings", return_value=settings):
                pipeline, lookup = script.load_pipeline("production")
            pipeline.dense._model = FakeEmbedder()
            pipeline.reranker = StubCrossEncoderReranker()
            self.assertIsInstance(pipeline.bm25, Bm25sRetriever)
            result = script.retrieve_all_systems(pipeline, lookup, QUESTION, depth=5)
            self.assertEqual(result["systems"]["bm25"]["chunk_ids"][0], "PHHC010000012020_chunk_0")
            self.assertEqual(result["systems"]["bm25"]["cnrs"][0], "PHHC010000012020")
            self.assertIn("Section 438", result["context"]["hybrid_rrf"][0]["text"])

            missing = settings_for(Path(tmp) / "empty", bm25_backend="bm25s")
            with mock.patch("legalrag.api.config.get_settings", return_value=missing):
                with self.assertRaises(SystemExit):
                    script.load_pipeline("production")


@unittest.skipUnless(HAVE_EXPORT_STACK and bm25s is not None, "needs torch, transformers, onnx, onnxruntime and bm25s")
class TestFullV2StackWithOnnxModels(unittest.TestCase):
    """bm25s + SQLite + IVF + ONNX embedder and reranker, from real exported tiny models."""

    @classmethod
    def setUpClass(cls):
        from legalrag.retrieval.onnx_export import export_embedder, export_reranker

        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name) / "v2"
        cls.dir.mkdir()
        make_tiny_model(Path(cls.tmp.name) / "emb_src", texts=TEXTS + [QUESTION])
        make_tiny_model(Path(cls.tmp.name) / "rr_src", labels=1, texts=TEXTS + [QUESTION])
        export_embedder(str(Path(cls.tmp.name) / "emb_src"), cls.dir / "onnx" / "embedder", pooling="cls")
        export_reranker(str(Path(cls.tmp.name) / "rr_src"), cls.dir / "onnx" / "reranker")
        embedder = OnnxEmbedder.load(cls.dir / "onnx" / "embedder")
        write_v2_artifacts(cls.dir, dense="ivf", embedder=embedder)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        reset_pipeline_cache()
        self.addCleanup(reset_pipeline_cache)

    def flags(self, **extra):
        return dict(bm25_backend="bm25s", chunk_store="sqlite", embedder_backend="onnx", reranker_backend="onnx",
                    dense_nprobe=2, dense_mmap=True, **extra)

    def test_loader_builds_the_onnx_models_and_warmup_passes(self):
        pipeline = load_production_pipeline(Settings(_env_file=None, environment="production", groq_api_key="gsk_dummy_key_for_test",
                                                     artifact_dir=self.dir, model_warmup_enabled=True, **self.flags()))
        self.assertIsInstance(pipeline.dense._model, OnnxEmbedder)
        self.assertIsInstance(pipeline.reranker, OnnxReranker)
        response = ask(pipeline)
        self.assertEqual(response.status, "ok")
        self.assertGreater(len(response.retrieved_chunk_ids), 0)
        self.assertTrue(all(c in CHUNK_IDS for c in response.retrieved_chunk_ids))
        self.assertTrue(all(c.chunk_id in response.retrieved_chunk_ids for c in response.citations))
        self.assertEqual(response.retrieved_chunk_ids[0] in CHUNK_IDS, True)

    def test_missing_onnx_directories_are_reported(self):
        import shutil

        copy = Path(self.tmp.name) / "no_onnx"
        shutil.copytree(self.dir, copy, ignore=shutil.ignore_patterns("onnx"))
        with self.assertRaises(FileNotFoundError) as ctx:
            load_production_pipeline(settings_for(copy, **self.flags()))
        self.assertIn("ONNX embedder directory missing", str(ctx.exception))
        self.assertIn("ONNX reranker directory missing", str(ctx.exception))

    def test_a_model_directory_of_the_wrong_kind_fails_validation(self):
        import shutil

        swapped = Path(self.tmp.name) / "swapped"
        shutil.copytree(self.dir, swapped)
        shutil.rmtree(swapped / "onnx" / "reranker")
        shutil.copytree(swapped / "onnx" / "embedder", swapped / "onnx" / "reranker")
        with self.assertRaisesRegex(ValueError, "'embedder' model but a 'reranker'"):
            load_production_pipeline(settings_for(swapped, **self.flags()))

    def test_the_whole_stack_answers_a_query_in_a_process_that_cannot_import_pytorch(self):
        code = f"""
import sys
for name in ("torch", "transformers", "sentence_transformers", "rank_bm25", "langchain", "langchain_community"):
    sys.modules[name] = None
import json
from pathlib import Path
from legalrag.api.config import Settings
from legalrag.api.dependencies import load_production_pipeline
from legalrag.api.routes import query_endpoint
from legalrag.api.schemas import QueryRequest
from legalrag.generation.client import GenerationResult
import re
class Echo:
    model_name = "echo"
    def generate(self, system_prompt, user_prompt, temperature=0.0, max_tokens=1024):
        ids = list(dict.fromkeys(re.findall(r"\\[Chunk ID:\\s*([^\\]|\\s][^\\]|]*?)\\s*(?:\\||\\])", user_prompt)))[:2]
        return GenerationResult(text="Held " + " ".join("[Chunk ID: %s]" % i for i in ids), status="success", model_name="echo")
settings = Settings(_env_file=None, environment="production", groq_api_key="gsk_dummy_key_for_test",
                    artifact_dir=Path({str(self.dir)!r}), model_warmup_enabled=True, bm25_backend="bm25s",
                    chunk_store="sqlite", embedder_backend="onnx", reranker_backend="onnx", dense_nprobe=2, dense_mmap=True)
pipeline = load_production_pipeline(settings)
pipeline.generator = Echo()
response = query_endpoint(QueryRequest.model_validate({{"question": {QUESTION!r}}}), pipeline)
loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("torch", "transformers") and sys.modules[m] is not None)
print(json.dumps({{"status": response.status, "ids": response.retrieved_chunk_ids, "citations": len(response.citations), "torch_loaded": loaded}}))
"""
        src = str(Path(__file__).resolve().parents[1] / "src")
        done = subprocess.run([sys.executable, "-c", code], env={"PYTHONPATH": src, "PATH": "", "ENVIRONMENT": "production"},
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        report = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual(report["status"], "ok")
        self.assertGreater(len(report["ids"]), 0)
        self.assertEqual(report["torch_loaded"], [])


if __name__ == "__main__":
    unittest.main()
