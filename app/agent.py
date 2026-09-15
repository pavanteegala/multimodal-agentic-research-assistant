"""
Agentic orchestration layer for the multimodal research assistant.

The ResearchAgent is responsible for:
    - understanding the question
    - selecting a retrieval strategy
    - retrieving text and/or visual evidence
    - handling multi-hop questions
    - generating grounded answers
    - resolving evidence references into citations
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


# ============================================================
# Evidence model
# ============================================================

@dataclass
class EvidenceItem:
    """
    Normalized representation of one retrieved evidence item.
    """

    evidence_id: str
    evidence_type: str
    metadata: dict[str, Any]
    score: float = 0.0

    @property
    def page(self) -> Any:
        return self.metadata.get("page", "unknown")

    @property
    def source(self) -> str:
        return str(
            self.metadata.get(
                "source",
                "unknown source",
            )
        )

    @property
    def text(self) -> str:
        return str(
            self.metadata.get(
                "text",
                self.metadata.get(
                    "analysis",
                    "",
                ),
            )
        )

    @property
    def image(self) -> str:
        return str(
            self.metadata.get(
                "image",
                self.metadata.get(
                    "filename",
                    "",
                ),
            )
        )


# ============================================================
# Research Agent
# ============================================================

class ResearchAgent:
    """
    Coordinates query planning, retrieval, reasoning, and citations.
    """

    def __init__(
        self,
        text_store: Any,
        visual_store: Any,
        query_detector: Callable[[str], str],
        query_embedder: Callable[[str], Any],
        text_searcher: Callable[..., list[dict]],
        visual_searcher: Callable[..., list[dict]],
        context_builder: Callable[..., str],
        answer_generator: Callable[..., str],
    ) -> None:

        self.text_store = text_store
        self.visual_store = visual_store

        self.query_detector = query_detector
        self.query_embedder = query_embedder

        self.text_searcher = text_searcher
        self.visual_searcher = visual_searcher

        self.context_builder = context_builder
        self.answer_generator = answer_generator

    # ========================================================
    # General helpers
    # ========================================================

    @staticmethod
    def _normalize(value: Any) -> str:
        """
        Normalize whitespace and casing.
        """

        return re.sub(
            r"\s+",
            " ",
            str(value).lower(),
        ).strip()

    @staticmethod
    def _tokens(value: Any) -> set[str]:
        """
        Extract simple alphanumeric tokens.
        """

        return set(
            re.findall(
                r"[a-z0-9]+",
                ResearchAgent._normalize(value),
            )
        )

    @staticmethod
    def _clean_sentence(value: str) -> str:
        """
        Clean one sentence of unnecessary whitespace.
        """

        return re.sub(
            r"\s+",
            " ",
            str(value),
        ).strip()

    # ========================================================
    # Query planning
    # ========================================================

    def plan(self, question: str) -> dict[str, Any]:
        """
        Determine the high-level strategy for a question.
        """

        normalized = self._normalize(question)
        query_type = self.query_detector(question)

        count_patterns = (
            "how many",
            "number of",
            "count of",
            "total number",
            "how much",
        )

        list_patterns = (
            "list",
            "which are",
            "what are",
            "name the",
            "give me the",
        )

        explain_patterns = (
            "why",
            "how does",
            "how did",
            "explain",
            "reason",
        )

        multi_hop_patterns = (
            " and ",
            " compared with ",
            " compared to ",
            " difference between ",
            " versus ",
            " vs ",
        )

        if any(pattern in normalized for pattern in count_patterns):
            task_type = "count"
        elif any(pattern in normalized for pattern in list_patterns):
            task_type = "list"
        elif any(pattern in normalized for pattern in multi_hop_patterns):
            task_type = "multi_hop"
        elif any(pattern in normalized for pattern in explain_patterns):
            task_type = "explain"
        else:
            task_type = "lookup"

        # Questions that explicitly combine multiple operations
        # should use the multi-hop route.
        if (
            ("why" in normalized and "how" in normalized)
            or (
                " and " in normalized
                and any(
                    word in normalized
                    for word in (
                        "why",
                        "compare",
                        "difference",
                    )
                )
            )
        ):
            task_type = "multi_hop"

        return {
            "task_type": task_type,
            "query_type": query_type,
            "question": question,
        }

    # ========================================================
    # Question decomposition
    # ========================================================

    def decompose_question(
        self,
        question: str,
    ) -> list[str]:
        """
        Break complex questions into smaller retrieval questions.

        This deliberately uses deterministic rules rather than
        another LLM call, keeping the pipeline fast and quota-aware.
        """

        original = self._clean_sentence(question)
        normalized = self._normalize(original)

        subquestions: list[str] = []

        # ----------------------------------------------------
        # Why + how pattern
        # ----------------------------------------------------

        if "why" in normalized and "how" in normalized:
            why_match = re.search(
                r"why\s+(.+?)(?:,\s*and\s+how|\s+and\s+how|\?)",
                original,
                flags=re.IGNORECASE,
            )

            how_match = re.search(
                r"how\s+(.+?)(?:\?|$)",
                original,
                flags=re.IGNORECASE,
            )

            if why_match:
                subquestions.append(
                    f"Why {why_match.group(1).strip()}?"
                )

            if how_match:
                subquestions.append(
                    f"How {how_match.group(1).strip()}?"
                )

        # ----------------------------------------------------
        # Comparison / difference pattern
        # ----------------------------------------------------

        comparison_match = re.search(
            r"(?:compare|difference between|compared with|compared to)\s+(.+)",
            original,
            flags=re.IGNORECASE,
        )

        if comparison_match:
            comparison_text = comparison_match.group(1).strip()
            comparison_text = comparison_text.rstrip("?.")

            parts = re.split(
                r"\s+(?:and|vs\.?|versus)\s+",
                comparison_text,
                flags=re.IGNORECASE,
            )

            if len(parts) >= 2:
                left = parts[0].strip()
                right = parts[1].strip()

                subquestions.extend(
                    [
                        f"What does the document say about {left}?",
                        f"What does the document say about {right}?",
                    ]
                )

        # ----------------------------------------------------
        # "X and Y" fallback
        # ----------------------------------------------------

        if not subquestions and " and " in normalized:
            parts = re.split(
                r"\s+and\s+",
                original,
                flags=re.IGNORECASE,
            )

            if 1 < len(parts) <= 3:
                for part in parts:
                    cleaned = part.strip(" .?")
                    if len(cleaned.split()) >= 3:
                        subquestions.append(
                            cleaned + "?"
                        )

        # ----------------------------------------------------
        # Final cleanup
        # ----------------------------------------------------

        cleaned_questions: list[str] = []

        for item in subquestions:
            item = self._clean_sentence(item)

            if not item:
                continue

            if not item.endswith("?"):
                item += "?"

            if item not in cleaned_questions:
                cleaned_questions.append(item)

        if not cleaned_questions:
            return [original]

        return cleaned_questions

    # ========================================================
    # Retrieval
    # ========================================================

    def retrieve_hop(
        self,
        question: str,
        query_type: str,
    ) -> dict[str, list[dict]]:
        """
        Retrieve evidence for one question/hop.
        """

        query_embedding = self.query_embedder(question)

        text_results: list[dict] = []
        visual_results: list[dict] = []

        # Text retrieval is available for all questions.
        try:
            text_results = self.text_searcher(
                question,
                query_embedding,
                self.text_store,
            )
        except Exception as error:
            print(
                "Text retrieval warning:",
                type(error).__name__,
                error,
            )

        # Visual retrieval is available for all questions,
        # but visual questions prioritize it later.
        try:
            visual_results = self.visual_searcher(
                question,
                query_embedding,
                self.visual_store,
            )
        except Exception as error:
            print(
                "Visual retrieval warning:",
                type(error).__name__,
                error,
            )

        return {
            "text": text_results,
            "visual": visual_results,
        }

    # ========================================================
    # Evidence normalization
    # ========================================================

    def _normalize_evidence(
        self,
        results: list[dict],
        evidence_type: str,
        start_index: int,
    ) -> list[EvidenceItem]:
        """
        Convert raw retrieval results into EvidenceItem objects.
        """

        normalized: list[EvidenceItem] = []

        for offset, result in enumerate(results):
            metadata = result.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {
                    "text": str(metadata)
                }

            normalized.append(
                EvidenceItem(
                    evidence_id=f"E{start_index + offset}",
                    evidence_type=evidence_type,
                    metadata=metadata,
                    score=float(
                        result.get(
                            "score",
                            0.0,
                        )
                    ),
                )
            )

        return normalized

    # ========================================================
    # Evidence selection
    # ========================================================

    def select_sources(
        self,
        query: str,
        query_type: str,
        text_results: list[dict],
        visual_results: list[dict],
    ) -> list[EvidenceItem]:
        """
        Select the most useful evidence.

        Visual questions prioritize visual evidence.
        Simple questions usually use one strong source.
        Multi-hop questions can use several sources.
        """

        plan = self.plan(query)
        task_type = plan["task_type"]

        selected: list[EvidenceItem] = []

        # ----------------------------------------------------
        # Visual questions
        # ----------------------------------------------------

        if query_type == "visual":
            selected.extend(
                self._normalize_evidence(
                    visual_results[:1],
                    "visual",
                    1,
                )
            )

            if not selected:
                selected.extend(
                    self._normalize_evidence(
                        text_results[:1],
                        "text",
                        1,
                    )
                )

            return selected

        # ----------------------------------------------------
        # Multi-hop questions
        # ----------------------------------------------------

        if task_type == "multi_hop":
            selected.extend(
                self._normalize_evidence(
                    text_results[:2],
                    "text",
                    1,
                )
            )

            visual_start = len(selected) + 1

            selected.extend(
                self._normalize_evidence(
                    visual_results[:1],
                    "visual",
                    visual_start,
                )
            )

            return selected

        # ----------------------------------------------------
        # Normal text questions
        # ----------------------------------------------------

        selected.extend(
            self._normalize_evidence(
                text_results[:1],
                "text",
                1,
            )
        )

        # Use visual evidence as a fallback or supporting source.
        if not selected:
            selected.extend(
                self._normalize_evidence(
                    visual_results[:1],
                    "visual",
                    1,
                )
            )

        return selected

    # ========================================================
    # Evidence context
    # ========================================================

    def build_evidence_context(
        self,
        evidence: list[EvidenceItem],
    ) -> str:
        """
        Build the tagged evidence context used by the LLM.
        """

        if not evidence:
            return "No relevant evidence was found."

        sections: list[str] = []

        for item in evidence:
            if item.evidence_type == "visual":
                sections.append(
                    f"[{item.evidence_id}]\n"
                    f"Type: Visual\n"
                    f"Source: {item.source}\n"
                    f"Page: {item.page}\n"
                    f"Image: {item.image}\n"
                    f"Analysis: {item.text}"
                )
            else:
                chunk_id = item.metadata.get(
                    "chunk_id",
                    "unknown",
                )

                sections.append(
                    f"[{item.evidence_id}]\n"
                    f"Type: Text\n"
                    f"Source: {item.source}\n"
                    f"Page: {item.page}\n"
                    f"Chunk: {chunk_id}\n"
                    f"Text: {item.text}"
                )

        return "\n\n".join(sections)

    # ========================================================
    # Fast local answers
    # ========================================================

    def fast_count_answer(
        self,
        question: str,
        evidence: list[EvidenceItem],
    ) -> str | None:
        """
        Attempt a lightweight local answer for straightforward
        count/number questions.

        Returns None when a reliable local answer cannot be formed.
        """

        normalized = self._normalize(question)

        count_question = (
            "how many" in normalized
            or "number of" in normalized
            or "count of" in normalized
            or "total number" in normalized
            or "how much" in normalized
        )

        if not count_question:
            return None

        candidates: list[str] = []

        for item in evidence:
            text = item.text

            # Sentences containing numbers are useful candidates.
            sentences = re.split(
                r"(?<=[.!?])\s+",
                text,
            )

            for sentence in sentences:
                sentence = self._clean_sentence(sentence)

                if re.search(
                    r"\d",
                    sentence,
                ):
                    candidates.append(sentence)

        if not candidates:
            return None

        # Prefer a sentence that contains terms from the question.
        question_tokens = self._tokens(question)

        scored: list[tuple[float, str]] = []

        for sentence in candidates:
            sentence_tokens = self._tokens(sentence)

            overlap = len(
                question_tokens & sentence_tokens
            )

            number_count = len(
                re.findall(
                    r"\d+(?:\.\d+)?",
                    sentence,
                )
            )

            score = (
                overlap * 2
                + min(number_count, 3)
            )

            scored.append(
                (score, sentence)
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        best_sentence = scored[0][1]

        return best_sentence

    # ========================================================
    # Multi-hop answering
    # ========================================================

    def multi_hop_answer(
        self,
        question: str,
        hop_contexts: list[str],
    ) -> str:
        """
        Ask the answer generator to synthesize multiple
        retrieval hops into one grounded response.
        """

        combined_context = "\n\n".join(
            hop_contexts
        )

        prompt = f"""
You are a research assistant answering a multi-step question.

User question:
{question}

Retrieved evidence:
{combined_context}

Instructions:
1. Reason only from the retrieved evidence.
2. Answer every part of the user's question.
3. Do not invent unsupported information.
4. Preserve evidence tags such as [E1], [E2], [E3] when making claims.
5. Keep the answer clear and concise.
"""

        return self.answer_generator(
            question,
            "multi_hop",
            prompt,
        )

    # ========================================================
    # Citation resolution
    # ========================================================

    def resolve_citations(
        self,
        answer: str,
        evidence: list[EvidenceItem],
    ) -> tuple[str, list[dict]]:
        """
        Convert [E1], [E2], etc. into source metadata.
        """

        evidence_map = {
            item.evidence_id: item
            for item in evidence
        }

        citation_pattern = re.compile(
            r"\[(E\d+)\]",
            flags=re.IGNORECASE,
        )

        citations: list[dict] = []
        seen_ids: set[str] = set()

        def replace_match(match: re.Match) -> str:
            evidence_id = match.group(1).upper()

            item = evidence_map.get(
                evidence_id
            )

            if item is None:
                return match.group(0)

            if evidence_id not in seen_ids:
                seen_ids.add(evidence_id)

                citation = {
                    "evidence_id": evidence_id,
                    "type": item.evidence_type,
                    "source": item.source,
                    "page": item.page,
                    "chunk_id": item.metadata.get(
                        "chunk_id"
                    ),
                    "image": item.image or None,
                    "metadata": item.metadata,
                }

                citations.append(citation)

            if item.evidence_type == "visual":
                if item.image:
                    return (
                        f"[Page {item.page}, "
                        f"Image: {item.image}]"
                    )

                return f"[Page {item.page}, Visual]"

            chunk_id = item.metadata.get(
                "chunk_id"
            )

            if chunk_id is not None:
                return (
                    f"[Page {item.page}, "
                    f"Chunk {chunk_id}]"
                )

            return f"[Page {item.page}]"

        resolved_answer = citation_pattern.sub(
            replace_match,
            answer,
        )

        # If Gemini did not emit evidence tags, still return
        # source metadata for the UI.
        if not citations:
            for item in evidence:
                citations.append(
                    {
                        "evidence_id": item.evidence_id,
                        "type": item.evidence_type,
                        "source": item.source,
                        "page": item.page,
                        "chunk_id": item.metadata.get(
                            "chunk_id"
                        ),
                        "image": item.image or None,
                        "metadata": item.metadata,
                    }
                )

        return resolved_answer, citations

    # ========================================================
    # Main agent pipeline
    # ========================================================

    def run(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        Execute the complete agent pipeline.
        """

        question = self._clean_sentence(question)

        if not question:
            return {
                "answer": "Please enter a question.",
                "citations": [],
                "plan": {},
            }

        # ----------------------------------------------------
        # 1. Plan
        # ----------------------------------------------------

        plan = self.plan(question)

        task_type = plan["task_type"]
        query_type = plan["query_type"]

        print(
            f"[Agent] task={task_type}, "
            f"query_type={query_type}"
        )

        # ----------------------------------------------------
        # 2. Multi-hop path
        # ----------------------------------------------------

        if task_type == "multi_hop":
            subquestions = self.decompose_question(
                question
            )

            print(
                f"[Agent] hops={len(subquestions)}"
            )

            hop_contexts: list[str] = []
            all_evidence: list[EvidenceItem] = []

            for hop_index, subquestion in enumerate(
                subquestions,
                start=1,
            ):
                print(
                    f"[Agent] Hop {hop_index}: "
                    f"{subquestion}"
                )

                results = self.retrieve_hop(
                    subquestion,
                    query_type,
                )

                evidence = self.select_sources(
                    subquestion,
                    query_type,
                    results["text"],
                    results["visual"],
                )

                hop_context = self.build_evidence_context(
                    evidence
                )

                hop_contexts.append(
                    f"Hop {hop_index}:\n{hop_context}"
                )

                all_evidence.extend(
                    evidence
                )

            # Re-number evidence IDs globally.
            renumbered_evidence: list[EvidenceItem] = []

            for index, item in enumerate(
                all_evidence,
                start=1,
            ):
                renumbered_evidence.append(
                    EvidenceItem(
                        evidence_id=f"E{index}",
                        evidence_type=item.evidence_type,
                        metadata=item.metadata,
                        score=item.score,
                    )
                )

            answer = self.multi_hop_answer(
                question,
                [
                    self.build_evidence_context(
                        [
                            EvidenceItem(
                                evidence_id=f"E{index}",
                                evidence_type=item.evidence_type,
                                metadata=item.metadata,
                                score=item.score,
                            )
                            for index, item in enumerate(
                                all_evidence,
                                start=1,
                            )
                        ]
                    )
                    for _ in [0]
                ],
            )

            resolved_answer, citations = (
                self.resolve_citations(
                    answer,
                    renumbered_evidence,
                )
            )

            return {
                "answer": resolved_answer,
                "citations": citations,
                "plan": plan,
                "subquestions": subquestions,
            }

        # ----------------------------------------------------
        # 3. Normal retrieval
        # ----------------------------------------------------

        results = self.retrieve_hop(
            question,
            query_type,
        )

        evidence = self.select_sources(
            question,
            query_type,
            results["text"],
            results["visual"],
        )

        # ----------------------------------------------------
        # 4. Fast local answer
        # ----------------------------------------------------

        local_answer = self.fast_count_answer(
            question,
            evidence,
        )

        if local_answer:
            resolved_answer, citations = (
                self.resolve_citations(
                    local_answer,
                    evidence,
                )
            )

            return {
                "answer": resolved_answer,
                "citations": citations,
                "plan": plan,
            }

        # ----------------------------------------------------
        # 5. Build answer context
        # ----------------------------------------------------

        context = self.build_evidence_context(
            evidence
        )

        # ----------------------------------------------------
        # 6. LLM answer generation
        # ----------------------------------------------------

        answer = self.answer_generator(
            question,
            query_type,
            context,
        )

        # ----------------------------------------------------
        # 7. Resolve citations
        # ----------------------------------------------------

        resolved_answer, citations = (
            self.resolve_citations(
                answer,
                evidence,
            )
        )

        return {
            "answer": resolved_answer,
            "citations": citations,
            "plan": plan,
        }