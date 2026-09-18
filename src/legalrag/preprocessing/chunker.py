"""
Recursive character text chunker configured with locked baseline hyperparameters
for Indian High Court judgments.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        # Fallback pure-Python recursive character text splitter
        class RecursiveCharacterTextSplitter:  # type: ignore
            def __init__(
                self,
                chunk_size: int = 1200,
                chunk_overlap: int = 200,
                separators: Optional[List[str]] = None,
            ):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = separators or ["\n\n", "\n", " ", ""]

            def split_text(self, text: str) -> List[str]:
                if not text:
                    return []
                return self._split_text(text, self.separators)

            def _split_text(self, text: str, separators: List[str]) -> List[str]:
                final_chunks: List[str] = []
                separator = separators[-1]
                new_separators = []
                for i, _s in enumerate(separators):
                    if _s == "":
                        separator = _s
                        break
                    if _s in text:
                        separator = _s
                        new_separators = separators[i + 1 :]
                        break

                splits = text.split(separator) if separator else list(text)
                good_splits: List[str] = []
                _sep = separator if separator else ""

                for s in splits:
                    if len(s) < self.chunk_size:
                        good_splits.append(s)
                    else:
                        if good_splits:
                            merged = self._merge_splits(good_splits, _sep)
                            final_chunks.extend(merged)
                            good_splits = []
                        if not new_separators:
                            final_chunks.append(s)
                        else:
                            other_chunks = self._split_text(s, new_separators)
                            final_chunks.extend(other_chunks)
                if good_splits:
                    merged = self._merge_splits(good_splits, _sep)
                    final_chunks.extend(merged)
                return final_chunks

            def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
                docs: List[str] = []
                current_doc: List[str] = []
                total = 0
                for s in splits:
                    _len = len(s)
                    if total + _len + (len(separator) if current_doc else 0) > self.chunk_size:
                        if total > 0:
                            doc = separator.join(current_doc).strip()
                            if doc:
                                docs.append(doc)
                            while total > self.chunk_overlap or (
                                total + _len + (len(separator) if current_doc else 0) > self.chunk_size and total > 0
                            ):
                                popped = current_doc.pop(0)
                                total -= len(popped) + (len(separator) if current_doc else 0)
                    current_doc.append(s)
                    total += _len + (len(separator) if len(current_doc) > 1 else 0)
                doc = separator.join(current_doc).strip()
                if doc:
                    docs.append(doc)
                return docs


@dataclass(frozen=True)
class Chunk:
    """Represents an atomic text chunk from a legal judgment."""

    chunk_id: str
    cnr: str
    chunk_index: int
    text: str
    court_name: Optional[str] = None
    decision_date: Optional[str] = None


class LegalChunker:
    """
    Splits legal judgment texts into cohesive context passages using
    the locked baseline hyperparameters:
        - chunk_size = 1200 characters (~200-250 tokens)
        - chunk_overlap = 200 characters (~35-40 tokens)
        - min_chunk_length = 100 characters
        - separators = ["\n\n", "\n", " ", ""]
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        min_chunk_length: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_text(self, text: str, cnr: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Splits a single judgment text into a list of Chunk objects.

        Args:
            text: Sanitized judgment full text.
            cnr: Unique case identifier.
            metadata: Optional additional metadata (e.g. court_name, decision_date).

        Returns:
            List of Chunk dataclass instances.
        """
        if not text or len(text.strip()) < self.min_chunk_length:
            return []

        court_name = metadata.get("court_name") if metadata else None
        decision_date = metadata.get("decision_date") if metadata else None

        raw_splits = self.splitter.split_text(text)
        chunks: List[Chunk] = []

        for idx, split in enumerate(raw_splits):
            cleaned_split = split.strip()
            if len(cleaned_split) >= self.min_chunk_length:
                chunk_id = f"{cnr}_chunk_{idx}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        cnr=cnr,
                        chunk_index=idx,
                        text=cleaned_split,
                        court_name=court_name,
                        decision_date=decision_date,
                    )
                )

        return chunks
