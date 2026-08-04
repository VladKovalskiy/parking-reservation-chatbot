"""PII detection and masking via Microsoft Presidio (ADR-005: English only).

Built entirely from Presidio's built-in English recognizers — no custom
recognizers needed:

- PERSON (and other NER-based entities) need a spaCy NER model, so
  `en_core_web_sm` is pinned as a project dependency (see pyproject.toml). We
  still start from Presidio's own `conf/default.yaml` (via a bare
  `NlpEngineProvider()`) and only swap the model name, rather than hand-
  rolling the NLP config from scratch — that file also carries the
  `labels_to_ignore` filter (drops noisy spaCy NER labels like ORGANIZATION,
  MONEY, ...) and the entity-name mapping; without it, clean text like
  "Reservations require confirmation" gets false-positive-tagged. Starting
  bare also avoids the default `AnalyzerEngine()` silently downloading the
  much larger `en_core_web_lg` (~400 MB) over the network on first use.
- Car-plate detection uses Presidio's UK vehicle-registration recognizer.
  It ships `enabled: false` in Presidio's own default recognizer list (it's
  opt-in even within the `uk` country filter), so it's added explicitly
  here — still an English-language recognizer, not a custom one (ADR-005).

Detection is scoped to exactly the four entity types this project actually
needs (`PII_ENTITIES` below) via `analyze(entities=...)`, not "whatever the
registry happens to support." Leaving it unscoped pulled in `DATE_TIME`
(among others) as a live entity, and `DATE_TIME` free-associates on totally
benign text with no PII in it at all — "today" alone scores 0.85. That
caused two separate live bugs before entities were scoped: "Working hours?"
got masked to "<DATE_TIME>?", losing the word `rag/router.py`'s classifier
keys on and misrouting the question to RAG; and "How are u today?" came
back from the LLM as "...help you with <DATE_TIME>?", leaking the
placeholder token itself into a normal reply. See CLAUDE.md's Known traps.
"""

from functools import lru_cache

from langchain_core.documents import Document
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import UkVehicleRegistrationRecognizer
from presidio_analyzer.recognizer_result import RecognizerResult
from presidio_anonymizer import AnonymizerEngine

from parking_bot.config import Settings, get_settings

SPACY_MODEL = "en_core_web_sm"

# Persons, phones, emails, car plates — see docs/evaluation.md-adjacent DOD
# for guardrails. Nothing else (DATE_TIME, LOCATION, NRP, URL, CREDIT_CARD,
# ...) is ever in scope, so it's never even asked for.
PII_ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "UK_VEHICLE_REGISTRATION"]


@lru_cache
def _analyzer() -> AnalyzerEngine:
    """Build the Presidio analyzer once per process (model load is expensive)."""
    provider = NlpEngineProvider()
    provider.nlp_configuration["models"] = [{"lang_code": "en", "model_name": SPACY_MODEL}]
    nlp_engine = provider.create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=["en"], nlp_engine=nlp_engine)
    registry.add_recognizer(UkVehicleRegistrationRecognizer())

    return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=["en"])


@lru_cache
def _anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


def detect_pii(text: str, *, settings: Settings | None = None) -> list[RecognizerResult]:
    """Return PII entities (`PII_ENTITIES`) found in `text` scoring above the threshold."""
    settings = settings or get_settings()
    return _analyzer().analyze(
        text=text,
        language="en",
        entities=PII_ENTITIES,
        score_threshold=settings.pii_score_threshold,
    )


def mask_pii(text: str, *, settings: Settings | None = None) -> str:
    """Replace detected PII spans in `text` with `<ENTITY_TYPE>` placeholders.

    A no-op when `settings.pii_enabled` is False or nothing is detected, so
    clean text is returned unchanged rather than passed through Presidio's
    anonymizer for nothing.
    """
    settings = settings or get_settings()
    if not settings.pii_enabled:
        return text
    results = detect_pii(text, settings=settings)
    if not results:
        return text
    return _anonymizer().anonymize(text=text, analyzer_results=results).text


def filter_documents(
    documents: list[Document], *, settings: Settings | None = None
) -> list[Document]:
    """Mask PII in retrieved chunk content before it leaves the vector store.

    Meant to sit between retrieval and prompt-building so PII that ended up
    in an indexed document (input-side guardrails should normally prevent
    that, but defense in depth) never reaches the LLM prompt or the user.
    """
    settings = settings or get_settings()
    return [
        Document(page_content=mask_pii(doc.page_content, settings=settings), metadata=doc.metadata)
        for doc in documents
    ]
