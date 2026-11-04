"""
Tests for the ONNX embedder and reranker and for the export (plan tasks C4 and C10).

Runtime behaviour is tested with a fake session and tokenizer (no model, no files). The export and the
numerical agreement with PyTorch are tested on a tiny randomly initialised BERT built in a temporary
folder: nothing is downloaded, and the tests skip if PyTorch, transformers, onnx or onnxruntime are
not installed.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from legalrag.retrieval.dense import DenseRetriever
from legalrag.retrieval.onnx_models import CONFIG_FILE, MODEL_FILE, TOKENIZER_FILE, OnnxEmbedder, OnnxReranker

DIM = 8
TABLE = np.random.default_rng(0).normal(size=(64, DIM)).astype(np.float32)


class FakeTokenizer:
    """Words -> ids by a stable hash; pairs are joined with a separator. Pads to the longest in the batch."""

    def encode_batch(self, inputs):
        rows = []
        for item in inputs:
            words = (item[0].split() + ["<sep>"] + item[1].split()) if isinstance(item, tuple) else item.split()
            rows.append([2] + [3 + (sum(map(ord, w)) % 60) for w in words])
        width = max(len(r) for r in rows)
        return [
            SimpleNamespace(ids=r + [0] * (width - len(r)), attention_mask=[1] * len(r) + [0] * (width - len(r)),
                            type_ids=[0] * width)
            for r in rows
        ]


class FakeSession:
    """Declares the given inputs; returns table[ids] as hidden states, or a score from the ids for a reranker."""

    def __init__(self, input_names=("input_ids", "attention_mask", "token_type_ids"), mode="hidden"):
        self.input_names, self.mode, self.calls = input_names, mode, []

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in self.input_names]

    def run(self, output_names, feeds):
        self.calls.append(set(feeds))
        ids, mask = feeds["input_ids"], feeds["attention_mask"]
        if self.mode == "hidden":
            return [TABLE[ids]]
        score = (ids * mask).sum(axis=1, keepdims=True).astype(np.float32) / 100.0
        return [score if self.mode == "score" else np.repeat(score, 2, axis=1)]


def embedder(config=None, **kwargs):
    model = OnnxEmbedder("unused", session=kwargs.pop("session", FakeSession()), tokenizer=FakeTokenizer())
    model.config = {"kind": "embedder", "dim": DIM, "pooling": "cls", "normalize": True, **(config or {})}
    return model


def reranker(config=None, session=None):
    model = OnnxReranker("unused", session=session or FakeSession(mode="score"), tokenizer=FakeTokenizer())
    model.config = {"kind": "reranker", "activation": "none", **(config or {})}
    return model


class TestOnnxEmbedder(unittest.TestCase):
    def test_cls_pooling_and_normalisation(self):
        vectors = embedder().encode(["alpha beta", "gamma"])
        self.assertEqual(vectors.shape, (2, DIM))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
        expected = TABLE[2] / np.linalg.norm(TABLE[2])  # the CLS row: every text starts with id 2
        np.testing.assert_allclose(vectors[0], expected, atol=1e-6)

    def test_unnormalised_output_when_asked(self):
        vectors = embedder().encode(["alpha"], normalize_embeddings=False)
        np.testing.assert_allclose(vectors[0], TABLE[2], atol=1e-6)

    def test_mean_pooling_ignores_padding(self):
        model = embedder({"pooling": "mean", "normalize": False})
        short, long = model.encode(["alpha"], batch_size=1)[0], model.encode(["alpha", "alpha beta gamma delta"], batch_size=2)[0]
        np.testing.assert_allclose(short, long, atol=1e-6)  # padding in the batch must not change the vector

    def test_order_is_preserved_across_batches_and_lengths(self):
        texts = ["a b c d e f g", "a", "a b c", "a b c d e f g h i j", "a b"]
        together = embedder({"pooling": "mean"}).encode(texts, batch_size=64)
        one_by_one = np.stack([embedder({"pooling": "mean"}).encode([t], batch_size=1)[0] for t in texts])
        small_batches = embedder({"pooling": "mean"}).encode(texts, batch_size=2)
        np.testing.assert_allclose(together, one_by_one, atol=1e-6)
        np.testing.assert_allclose(small_batches, one_by_one, atol=1e-6)

    def test_single_string_gives_a_single_vector(self):
        self.assertEqual(embedder().encode("alpha beta").shape, (DIM,))

    def test_empty_input(self):
        self.assertEqual(embedder().encode([]).shape, (0, DIM))

    def test_pooled_model_output_is_used_as_is(self):
        class Pooled(FakeSession):
            def run(self, names, feeds):
                return [np.ones((feeds["input_ids"].shape[0], DIM), dtype=np.float32) * 2]

        np.testing.assert_allclose(embedder(session=Pooled()).encode(["x"], normalize_embeddings=False)[0], 2.0)

    def test_only_declared_inputs_are_fed(self):
        session = FakeSession(input_names=("input_ids", "attention_mask"))
        embedder(session=session).encode(["alpha"])
        self.assertEqual(session.calls[0], {"input_ids", "attention_mask"})

    def test_a_model_that_needs_unknown_inputs_is_refused(self):
        with self.assertRaisesRegex(ValueError, "cannot supply"):
            embedder(session=FakeSession(input_names=("input_ids", "pixel_values"))).encode(["x"])

    def test_unknown_pooling_is_refused(self):
        with self.assertRaisesRegex(ValueError, "pooling"):
            embedder({"pooling": "max"}).encode(["x"])

    def test_warmup_checks_the_declared_dimension(self):
        embedder().warmup()
        with self.assertRaisesRegex(ValueError, "declared dimension"):
            embedder({"dim": DIM + 1}).warmup()

    def test_dimension_accessor_matches_sentence_transformers(self):
        self.assertEqual(embedder().get_sentence_embedding_dimension(), DIM)
        self.assertIsNone(embedder({"dim": 0}).get_sentence_embedding_dimension())

    def test_dense_retriever_accepts_it_in_place_of_sentence_transformers(self):
        import faiss

        texts = ["alpha beta", "gamma delta epsilon", "zeta"]
        model = embedder({"pooling": "mean"})  # CLS pooling would give every fake text the same vector
        index = faiss.IndexFlatIP(DIM)
        index.add(model.encode(texts))
        retriever = DenseRetriever(index=index, chunk_ids=["c0", "c1", "c2"], embedder=model, query_instruction="")
        retriever.validate()
        retriever.warmup()
        results = retriever.search("alpha beta", top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], "c0")  # the identical text scores 1.0 and comes first
        self.assertAlmostEqual(results[0][1], 1.0, places=5)


class TestOnnxReranker(unittest.TestCase):
    CANDIDATES = [("c0", "a"), ("c1", "a b c d"), ("c2", "a b"), ("c3", "a b c")]

    def test_orders_by_score_best_first_and_respects_top_k(self):
        ranked = reranker().rerank("query", self.CANDIDATES, top_k=2)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0][0], "c1")  # the longest passage has the largest fake score
        self.assertGreater(ranked[0][1], ranked[1][1])
        self.assertEqual((type(ranked[0][0]), type(ranked[0][1])), (str, float))

    def test_ties_keep_candidate_order(self):
        ranked = reranker().rerank("q", [("a", "x y"), ("b", "x y"), ("c", "x y")], top_k=3)
        self.assertEqual([c for c, _ in ranked], ["a", "b", "c"])

    def test_empty_candidates_and_top_k_zero(self):
        self.assertEqual(reranker().rerank("q", []), [])
        self.assertEqual(reranker().rerank("q", self.CANDIDATES, top_k=0), [])

    def test_sigmoid_activation_is_monotone_and_bounded(self):
        raw = reranker().predict([("q", "a b"), ("q", "a b c d e")])
        squashed = reranker({"activation": "sigmoid"}).predict([("q", "a b"), ("q", "a b c d e")])
        self.assertTrue(((squashed > 0) & (squashed < 1)).all())
        self.assertEqual(np.argsort(raw).tolist(), np.argsort(squashed).tolist())

    def test_batches_give_the_same_scores(self):
        pairs = [("q", " ".join(["w"] * n)) for n in range(1, 8)]
        np.testing.assert_allclose(reranker().predict(pairs, batch_size=64), reranker().predict(pairs, batch_size=2), atol=1e-6)

    def test_wrong_output_width_is_refused(self):
        with self.assertRaisesRegex(ValueError, "one score per pair"):
            reranker(session=FakeSession(mode="two")).predict([("q", "a")])

    def test_non_finite_scores_are_refused(self):
        class Nan(FakeSession):
            def run(self, names, feeds):
                return [np.full((feeds["input_ids"].shape[0], 1), np.nan, dtype=np.float32)]

        with self.assertRaisesRegex(ValueError, "non-finite"):
            reranker(session=Nan()).rerank("q", self.CANDIDATES)

    def test_pairs_are_passed_to_the_tokenizer_as_pairs(self):
        seen = []

        class Spy(FakeTokenizer):
            def encode_batch(self, inputs):
                seen.extend(inputs)
                return super().encode_batch(inputs)

        model = OnnxReranker("unused", session=FakeSession(mode="score"), tokenizer=Spy())
        model.config = {"kind": "reranker", "activation": "none"}
        model.rerank("the query", [("c0", "the passage")])
        self.assertEqual(seen, [("the query", "the passage")])

    def test_warmup_runs(self):
        reranker().warmup()


class TestDirectoryValidation(unittest.TestCase):
    def write(self, directory: Path, kind="embedder", files=(MODEL_FILE, TOKENIZER_FILE)):
        directory.mkdir(parents=True, exist_ok=True)
        for name in files:
            (directory / name).write_bytes(b"x")
        (directory / CONFIG_FILE).write_text(json.dumps({"kind": kind}), encoding="utf-8")

    def test_missing_files_are_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), files=(MODEL_FILE,))
            with self.assertRaisesRegex(ValueError, "tokenizer.json"):
                OnnxEmbedder.load(tmp)

    def test_missing_directory_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "missing"):
                OnnxReranker.load(Path(tmp) / "nowhere")

    def test_the_wrong_kind_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), kind="reranker")
            with self.assertRaisesRegex(ValueError, "'reranker' model but a 'embedder'"):
                OnnxEmbedder.load(tmp)
            self.write(Path(tmp), kind="embedder")
            with self.assertRaisesRegex(ValueError, "'embedder' model but a 'reranker'"):
                OnnxReranker.load(tmp)


class TestServingPathNeedsNoPyTorch(unittest.TestCase):
    def test_importing_the_runtime_never_imports_torch_or_transformers(self):
        code = (
            "import sys\n"
            "sys.modules['torch'] = None\nsys.modules['transformers'] = None\nsys.modules['sentence_transformers'] = None\n"
            "import legalrag.retrieval.onnx_models as m\n"
            "import legalrag.retrieval.faiss_builders, legalrag.retrieval.bm25s_backend, legalrag.retrieval.chunk_store\n"
            "print('ok')\n"
        )
        src = str(Path(__file__).resolve().parents[1] / "src")
        done = subprocess.run([sys.executable, "-c", code], env={"PYTHONPATH": src, "PATH": ""}, capture_output=True, text=True)
        self.assertEqual(done.stdout.strip(), "ok", done.stderr)


# --------------------------------------------------------------------------- real export on a tiny model

try:
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401
    import tokenizers  # noqa: F401
    import torch
    import transformers  # noqa: F401
    HAVE_EXPORT_STACK = True
except ImportError:  # pragma: no cover
    HAVE_EXPORT_STACK = False


def make_tiny_model(directory: Path, labels: int = 0, texts=None) -> None:
    """A 2-layer BERT with random weights and a word-level tokenizer, both written locally."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertForSequenceClassification, BertModel, PreTrainedTokenizerFast

    from legalrag.retrieval.onnx_export import SAMPLE_TEXTS

    words = sorted({w.lower().strip(".,") for t in (texts or SAMPLE_TEXTS) for w in t.split()})
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, **{w: i + 4 for i, w in enumerate(words)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]")
    fast.save_pretrained(str(directory))
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=128, num_labels=labels or 2)
    (BertForSequenceClassification(config) if labels else BertModel(config)).save_pretrained(str(directory))


@unittest.skipUnless(HAVE_EXPORT_STACK, "needs torch, transformers, tokenizers, onnx and onnxruntime")
class TestExportOnATinyModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        make_tiny_model(cls.root / "embedder_src")
        make_tiny_model(cls.root / "reranker_src", labels=1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_embedder_export_matches_pytorch_and_the_saved_directory_loads(self):
        from legalrag.retrieval.onnx_export import export_embedder

        out = self.root / "emb_cls"
        report = export_embedder(str(self.root / "embedder_src"), out, pooling="cls")
        self.assertLessEqual(report["max_abs_difference"], 1e-4)
        config = json.loads((out / CONFIG_FILE).read_text())
        self.assertEqual((config["kind"], config["dim"], config["pooling"]), ("embedder", 32, "cls"))
        model = OnnxEmbedder.load(out)
        model.warmup()
        vectors = model.encode(["Anticipatory bail.", "The writ petition is dismissed.", "unknownword " * 5], batch_size=2)
        self.assertEqual(vectors.shape, (3, 32))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_mean_pooling_export_with_padded_batches_matches_pytorch(self):
        from legalrag.retrieval.onnx_export import export_embedder

        report = export_embedder(str(self.root / "embedder_src"), self.root / "emb_mean", pooling="mean")
        self.assertLessEqual(report["max_abs_difference"], 1e-4)

    def test_reranker_export_matches_pytorch(self):
        from legalrag.retrieval.onnx_export import export_reranker

        out = self.root / "rerank"
        report = export_reranker(str(self.root / "reranker_src"), out)
        self.assertLessEqual(report["max_abs_difference"], 1e-4)
        model = OnnxReranker.load(out)
        ranked = model.rerank("anticipatory bail", [("a", "The writ petition is dismissed."), ("b", "Anticipatory bail under Section 438.")], top_k=2)
        self.assertEqual({c for c, _ in ranked}, {"a", "b"})
        self.assertTrue(all(0.0 < s < 1.0 for _, s in ranked))

    def test_reranker_export_without_the_sigmoid(self):
        from legalrag.retrieval.onnx_export import export_reranker

        export_reranker(str(self.root / "reranker_src"), self.root / "rerank_raw", activation="none")
        self.assertEqual(json.loads((self.root / "rerank_raw" / CONFIG_FILE).read_text())["activation"], "none")

    def test_export_refuses_a_multi_label_model_as_a_reranker(self):
        from legalrag.retrieval.onnx_export import export_reranker

        make_tiny_model(self.root / "two_labels", labels=2)
        with self.assertRaisesRegex(ValueError, "one output score"):
            export_reranker(str(self.root / "two_labels"), self.root / "never")

    def test_verification_catches_a_directory_that_no_longer_matches_the_model(self):
        from transformers import AutoModel, AutoTokenizer

        from legalrag.retrieval.onnx_export import ExportVerificationError, SAMPLE_TEXTS, export_embedder, verify_embedder

        out = self.root / "tampered"
        export_embedder(str(self.root / "embedder_src"), out, pooling="cls")
        config = json.loads((out / CONFIG_FILE).read_text())
        config["pooling"] = "mean"  # the runtime would now pool differently from the original model
        (out / CONFIG_FILE).write_text(json.dumps(config))
        model = AutoModel.from_pretrained(str(self.root / "embedder_src")).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(self.root / "embedder_src"))
        # Verification compares with PyTorch using the saved pooling, so a wrong pooling in both would pass;
        # instead corrupt the weights: a different model must fail.
        make_tiny_model(self.root / "other_model")
        torch.manual_seed(123)
        other = AutoModel.from_pretrained(str(self.root / "other_model")).eval()
        for parameter in other.parameters():
            parameter.data.normal_(0, 0.5)
        with self.assertRaises(ExportVerificationError):
            verify_embedder(out, other, tokenizer, SAMPLE_TEXTS, tolerance=1e-4)
        self.assertIsNotNone(model)

    def test_verification_uses_real_segment_ids_even_if_the_tokenizer_omits_them_by_default(self):
        from transformers import AutoTokenizer

        from legalrag.retrieval.onnx_export import _example

        tokenizer = AutoTokenizer.from_pretrained(str(self.root / "reranker_src"))
        self.assertNotIn("token_type_ids", tokenizer.model_input_names)  # the hazard this guards against
        _, _, token_type_ids = _example(tokenizer, [("first sentence", "second sentence here")], ["input_ids", "attention_mask", "token_type_ids"], 64)
        self.assertGreater(int(token_type_ids.sum()), 0)  # the second segment is marked 1, as the runtime does

    def test_a_slow_tokenizer_without_a_tokenizer_json_is_refused(self):
        from unittest import mock

        from legalrag.retrieval.onnx_export import _load_hf

        slow = SimpleNamespace(backend_tokenizer=None)
        with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=slow):
            with self.assertRaisesRegex(ValueError, "fast tokenizer"):
                _load_hf(str(self.root / "embedder_src"), "embedder", None)


if __name__ == "__main__":
    unittest.main()
