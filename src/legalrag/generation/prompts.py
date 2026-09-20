"""
Prompt templates and context formatting utilities for legal question answering.
"""

import re
from typing import List, Dict, Any, Optional

SYSTEM_PROMPT = """You are a specialized legal AI assistant analyzing Indian High Court judgments.

Your task is to answer legal inquiries strictly and faithfully based on the retrieved context passages provided below.

Strict Guidelines:
1. ONLY rely on factual assertions, statutory provisions, and judicial holdings directly stated in the context passages.
2. For every factual claim, legal provision, or procedural finding, explicitly cite the supporting passage using the format: [Chunk ID: <id>].
3. Do NOT invent citations, section numbers, case names, or judicial holdings.
4. If the retrieved passages do not contain sufficient evidence to answer the question, clearly state: "The provided judgment context does not contain sufficient information to answer this inquiry."
"""

# Matches both bare citations ("[Chunk ID: abc123]") and full context headers
# ("[Chunk ID: abc123 | Court: ... | Date: ...]") while capturing ONLY the chunk ID.
CITATION_PATTERN = re.compile(
    r"\[Chunk ID:\s*([^\]|\s][^\]|]*?)\s*(?:\||\])",
    re.IGNORECASE,
)


def format_context_block(
    chunk_id: str,
    text: str,
    court_code: Optional[str] = None,
    decision_date: Optional[str] = None,
) -> str:
    """Formats a single retrieved chunk into an attributed context block."""
    court_str = str(court_code) if court_code not in (None, "") else "Unspecified"
    date_str = decision_date if decision_date else "Unspecified"

    return f"[Chunk ID: {chunk_id} | Court: {court_str} | Date: {date_str}]\n{text}"


def build_rag_prompt(
    question: str,
    context_blocks: List[Dict[str, Any]],
) -> str:
    """
    Assembles retrieved context blocks and the user inquiry into a final prompt.

    Args:
        question: User inquiry or benchmark question.
        context_blocks: List of dicts with keys: chunk_id, text, (optional) court_code, decision_date.

    Returns:
        Formatted user message string.
    """
    formatted_passages = []
    for block in context_blocks:
        formatted = format_context_block(
            chunk_id=block["chunk_id"],
            text=block["text"],
            court_code=block.get("court_code"),
            decision_date=block.get("decision_date"),
        )
        formatted_passages.append(formatted)

    joined_context = "\n\n---\n\n".join(formatted_passages)

    return f"""### Retrieved Judgment Context Passages:
{joined_context}

### Legal Inquiry:
{question}

### Instructions:
Provide a precise, evidence-grounded answer based strictly on the context above, citing each supporting passage with [Chunk ID: ...]."""


def extract_citations(text: str) -> List[str]:
    """
    Extracts and deduplicates all [Chunk ID: ...] citations mentioned in an answer text.

    Args:
        text: Generated legal answer text.

    Returns:
        Ordered list of unique chunk IDs cited in the text.
    """
    if not text:
        return []

    found = CITATION_PATTERN.findall(text)
    seen = set()
    unique_citations = []
    for cid in found:
        cleaned_cid = cid.strip()
        if cleaned_cid and cleaned_cid not in seen:
            seen.add(cleaned_cid)
            unique_citations.append(cleaned_cid)

    return unique_citations
