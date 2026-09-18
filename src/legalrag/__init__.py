"""
LegalRAG: Hybrid Retrieval & Evidence-Grounded Question Answering
over Indian High Court Judgments.
"""

__version__ = "0.1.0"
__author__ = "Siddharth"

from legalrag.preprocessing.cleaner import clean_judgment_text, is_corrupted_document
from legalrag.preprocessing.chunker import LegalChunker, Chunk
from legalrag.retrieval.bm25 import BM25Retriever
from legalrag.retrieval.dense import DenseRetriever
from legalrag.retrieval.fusion import reciprocal_rank_fusion
from legalrag.retrieval.reranker import CrossEncoderReranker
from legalrag.evaluation.grounding import find_gold_chunks
from legalrag.evaluation.metrics import calculate_recall_at_k

__all__ = [
    "clean_judgment_text",
    "is_corrupted_document",
    "LegalChunker",
    "Chunk",
    "BM25Retriever",
    "DenseRetriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
    "find_gold_chunks",
    "calculate_recall_at_k",
]
