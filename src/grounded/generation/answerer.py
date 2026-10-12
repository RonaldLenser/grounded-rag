"""Question answering: retrieve → gate → reserve budget → LLM → verify → log.

The hallucination defenses, in order:
  1. Relevance gate: weak retrieval = refuse WITHOUT calling the LLM.
  2. Prompt contract: sources only, one claim per statement, sources are data.
  3. Structured output: an API-enforced JSON schema of claims with evidence.
  4. Claim verification in code (citations.py): the answer is built only from
     claims whose evidence verified. Unsupported claims are dropped, not edited.
  5. Every query logged (status, scores, tokens, latency) for evals and debugging.
"""

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from grounded.config import Settings
from grounded.db import repositories
from grounded.errors import DailyLimitError, LLMProviderError
from grounded.generation.citations import VerificationResult, verify_answer
from grounded.generation.prompts import (
    ANSWER_SCHEMA,
    INSUFFICIENT_CONTEXT_MESSAGE,
    SYSTEM_PROMPT,
    LLMAnswer,
    build_user_prompt,
    select_sources,
)
from grounded.providers.base import LLMClient
from grounded.retrieval.retriever import RetrievedChunk, Retriever

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AnswerResult:
    query_id: uuid.UUID
    verification: VerificationResult
    retrieved: list[RetrievedChunk]
    llm_called: bool
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int


class Answerer:
    def __init__(
        self,
        session: AsyncSession,
        retriever: Retriever,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self._session = session
        self._retriever = retriever
        self._llm = llm
        self._settings = settings

    async def answer(
        self, collection_id: uuid.UUID, question: str, top_k: int | None = None
    ) -> AnswerResult:
        started = time.perf_counter()
        chunks = await self._retriever.search(collection_id, question, top_k=top_k)

        model: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None

        if not chunks or chunks[0].score < self._settings.min_relevance_score:
            # Gate: nothing relevant enough. Refusing here costs $0 and can't hallucinate.
            verification = VerificationResult(
                status="insufficient_context", answer=INSUFFICIENT_CONTEXT_MESSAGE
            )
            llm_called = False
        else:
            await self._reserve_llm_call()
            sources = select_sources(chunks, self._settings.max_context_tokens)
            completion = await self._llm.complete_structured(
                system=SYSTEM_PROMPT,
                user=build_user_prompt(question, sources),
                output=ANSWER_SCHEMA,
                max_tokens=self._settings.llm_max_tokens,
            )
            try:
                llm_answer = LLMAnswer.model_validate(completion.data)
            except ValidationError as exc:
                logger.error("llm_output_invalid", errors=exc.error_count())
                raise LLMProviderError("The language model returned a malformed answer.") from exc
            verification = verify_answer(llm_answer, sources)
            llm_called = True
            model = completion.model
            input_tokens = completion.input_tokens
            output_tokens = completion.output_tokens

        latency_ms = round((time.perf_counter() - started) * 1000)
        query_log = await repositories.add_query_log(
            self._session,
            collection_id=collection_id,
            question=question,
            status=verification.status,
            retrieved_chunk_ids=[chunk.chunk_id for chunk in chunks],
            scores=[chunk.score for chunk in chunks],
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )
        await self._session.commit()

        # Counts and reasons only: never log the question, answer, or claim text.
        logger.info(
            "answer_completed",
            query_id=str(query_log.id),
            status=verification.status,
            llm_called=llm_called,
            top_score=round(chunks[0].score, 4) if chunks else None,
            claims_supported=len(verification.claims),
            claims_rejected=[c.reason for c in verification.rejected_claims],
            evidence_rejected=[e.reason for e in verification.rejected_evidence],
            downgraded_from=verification.downgraded_from,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )
        return AnswerResult(
            query_id=query_log.id,
            verification=verification,
            retrieved=chunks,
            llm_called=llm_called,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

    async def _reserve_llm_call(self) -> None:
        """Global cost ceiling: at most N LLM calls per UTC day, across all instances.

        The slot is reserved atomically in Postgres BEFORE the provider call and
        committed immediately, so:
          - concurrent requests can never overshoot the limit (no count-then-call race)
          - a call that fails or times out still counts (it may still be billed)
          - the counter's row lock is held for milliseconds, not for the LLM call
        Refusals at the relevance gate never get here: they cost nothing.
        """
        today = datetime.now(UTC).date()
        limit = self._settings.daily_llm_query_limit
        allowed = await repositories.reserve_llm_call(self._session, today, limit)
        await self._session.commit()
        if not allowed:
            logger.warning("daily_llm_limit_reached", limit=limit)
            raise DailyLimitError(
                "This demo has reached its daily question limit. Please try again tomorrow."
            )
