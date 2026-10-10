"""Generation quality metrics: Faithfulness, Relevance, Completeness, Hallucination."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GenerationMetrics:
    faithfulness: float
    answer_relevance: float
    completeness: float
    hallucination_score: float
    citation_present: bool
    response_length: int
    has_source_attribution: bool


def _sentence_split(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def faithfulness_score(answer: str, context_docs: list[dict]) -> float:
    """
    Measure how grounded the answer is in the provided context.
    Checks what fraction of answer sentences have supporting evidence in context.
    """
    if not answer or not context_docs:
        return 0.0

    context_text = " ".join(doc.get("text", "") for doc in context_docs).lower()
    sentences = _sentence_split(answer)
    if not sentences:
        return 0.0

    grounded = 0
    for sentence in sentences:
        # Check if key n-grams from the sentence appear in context
        words = sentence.lower().split()
        if len(words) < 3:
            grounded += 1
            continue

        # Check for overlapping 3-grams
        trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        matching = sum(1 for tg in trigrams if tg in context_text)
        if matching / max(len(trigrams), 1) >= 0.2:
            grounded += 1

    return round(grounded / len(sentences), 4)


def answer_relevance_score(answer: str, question: str) -> float:
    """
    Measure how relevant the answer is to the question.
    Uses keyword overlap between question and answer.
    """
    if not answer or not question:
        return 0.0

    # Extract meaningful words (skip stopwords)
    stopwords = {
        "what", "how", "does", "is", "are", "the", "a", "an", "in", "on", "of",
        "to", "for", "and", "or", "but", "not", "with", "from", "by", "at", "it",
        "this", "that", "these", "those", "do", "did", "was", "were", "be", "been",
        "about", "can", "will", "would", "should", "could", "has", "have", "had",
        "i", "you", "he", "she", "we", "they", "my", "your", "his", "her", "our",
        "their", "me", "him", "us", "them", "s", "say", "says", "said",
    }

    q_words = {w.lower().strip("?!.,") for w in question.split()} - stopwords
    a_words = {w.lower().strip("?!.,") for w in answer.split()} - stopwords

    if not q_words:
        return 1.0

    overlap = len(q_words & a_words)
    return round(min(overlap / len(q_words), 1.0), 4)


def completeness_score(answer: str, relevant_keywords: list[str]) -> float:
    """How many expected keywords are covered in the answer."""
    if not answer or not relevant_keywords:
        return 0.0

    answer_lower = answer.lower()
    covered = sum(1 for kw in relevant_keywords if kw.lower() in answer_lower)
    return round(covered / len(relevant_keywords), 4)


def hallucination_score(answer: str, context_docs: list[dict]) -> float:
    """
    Detect potential hallucinations: claims in the answer not supported by context.
    Returns ratio of unsupported sentences (lower is better).
    """
    if not answer:
        return 0.0
    if not context_docs:
        return 1.0

    context_text = " ".join(doc.get("text", "") for doc in context_docs).lower()
    sentences = _sentence_split(answer)
    if not sentences:
        return 0.0

    unsupported = 0
    for sentence in sentences:
        words = sentence.lower().split()
        if len(words) < 4:
            continue

        # Check for proper nouns, specific claims, numbers
        specific_patterns = re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b', sentence)
        if specific_patterns:
            found = sum(1 for p in specific_patterns if p.lower() in context_text)
            if found < len(specific_patterns) * 0.5:
                unsupported += 1
                continue

        # General n-gram grounding check
        trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        matching = sum(1 for tg in trigrams if tg in context_text)
        if trigrams and matching / len(trigrams) < 0.1:
            unsupported += 1

    return round(unsupported / max(len(sentences), 1), 4)


def check_citations(answer: str) -> bool:
    """Check if the answer includes source citations."""
    citation_patterns = [
        r'\[Source[:\s]',
        r'\[Source \d+',
        r'Source:',
        r'According to',
        r'Sadhguru (?:says|teaches|explains|describes|mentions)',
    ]
    return any(re.search(p, answer, re.IGNORECASE) for p in citation_patterns)


def compute_generation_metrics(
    answer: str,
    question: str,
    context_docs: list[dict],
    relevant_keywords: list[str],
) -> GenerationMetrics:
    """Compute all generation metrics for a single query."""
    faith = faithfulness_score(answer, context_docs)
    relevance = answer_relevance_score(answer, question)
    complete = completeness_score(answer, relevant_keywords)
    hallucination = hallucination_score(answer, context_docs)
    has_citation = check_citations(answer)

    return GenerationMetrics(
        faithfulness=faith,
        answer_relevance=relevance,
        completeness=complete,
        hallucination_score=hallucination,
        citation_present=has_citation,
        response_length=len(answer),
        has_source_attribution=has_citation,
    )
