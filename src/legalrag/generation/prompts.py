"""
Prompt templates and context formatting utilities for legal question answering.
"""

from typing import List, Dict, Any, Optional

SYSTEM_PROMPT = """You are a specialized legal AI assistant analyzing Indian High Court judgments.

Your task is to answer legal inquiries strictly and faithfully based on the retrieved context passages provided below.

Strict Guidelines:
1. ONLY rely on factual assertions, statutory provisions, and judicial holdings directly stated in the context passages.
2. For every factual claim, legal provision, or procedural finding, explicitly cite the supporting passage using the format: [Chunk ID: <id>].
3. Do NOT invent citations, section numbers, case names, or judicial holdings.
4. If the retrieved passages do not contain sufficient evidence to answer the question, clearly state: "The provided judgment context does not contain sufficient information to answer this inquiry."
"""


def format_context_block(
    chunk_id: str,
    text: str,
    court_name: Optional[str] = None,
    decision_date: Optional[str] = None,
) -> str:
    """Formats a single retrieved chunk into an attributed context block."""
    court_str = court_name if court_name else "Indian High Court"
    date_str = decision_date if decision_date else "Unspecified Date"

    return f"[Chunk ID: {chunk_id} | Court: {court_str} | Date: {date_str}]\n{text}"


def build_rag_prompt(
    question: str,
    context_blocks: List[Dict[str, Any]],
) -> str:
    """
    Assembles retrieved context blocks and the user inquiry into a final prompt.

    Args:
        question: User inquiry or benchmark question.
        context_blocks: List of dicts with keys: chunk_id, text, (optional) court_name, decision_date.

    Returns:
        Formatted user message string.
    """
    formatted_passages = []
    for block in context_blocks:
        formatted = format_context_block(
            chunk_id=block["chunk_id"],
            text=block["text"],
            court_name=block.get("court_name"),
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
