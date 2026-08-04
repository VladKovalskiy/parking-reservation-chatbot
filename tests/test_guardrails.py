from langchain_core.documents import Document

from parking_bot.config import Settings
from parking_bot.guardrails.pii import filter_documents, mask_pii

KNOWN_PII_TEXT = (
    "My name is John Smith, my phone number is (212) 555-0198, "
    "email john.smith@example.com, plate AB12 CDE."
)
CLEAN_TEXT = "Reservations require explicit confirmation before booking."


def test_mask_pii_detects_and_masks_known_pii_entities(settings: Settings) -> None:
    masked = mask_pii(KNOWN_PII_TEXT, settings=settings)

    assert "John Smith" not in masked
    assert "555-0198" not in masked
    assert "john.smith@example.com" not in masked
    assert "AB12 CDE" not in masked
    assert "<PERSON>" in masked
    assert "<PHONE_NUMBER>" in masked
    assert "<EMAIL_ADDRESS>" in masked
    assert "<UK_VEHICLE_REGISTRATION>" in masked


def test_mask_pii_passes_through_clean_text_unchanged(settings: Settings) -> None:
    assert mask_pii(CLEAN_TEXT, settings=settings) == CLEAN_TEXT


def test_mask_pii_does_not_touch_out_of_scope_entity_types(settings: Settings) -> None:
    """Regression test: detection is scoped to PII_ENTITIES, not "whatever
    Presidio's registry supports."

    Left unscoped, DATE_TIME free-associates on totally benign text with no
    PII at all — "today" alone scores 0.85 — which broke question routing
    ("Working hours?" -> "<DATE_TIME>?", losing the word the classifier
    keys on) and leaked the placeholder token into a normal chat reply
    ("...help you with <DATE_TIME>?") before entities were scoped. Neither
    text below contains PERSON/PHONE_NUMBER/EMAIL_ADDRESS/
    UK_VEHICLE_REGISTRATION, so both must pass through unchanged.
    """
    assert mask_pii("Hello! How are u today?", settings=settings) == "Hello! How are u today?"
    assert mask_pii("Working hours?", settings=settings) == "Working hours?"


def test_pii_score_threshold_is_configurable(settings: Settings) -> None:
    lenient = settings.model_copy(update={"pii_score_threshold": 0.5})
    strict = settings.model_copy(update={"pii_score_threshold": 0.99})

    # PHONE_NUMBER scores 0.75 here (base 0.4 + context boost from "phone
    # number"), so a 0.99 threshold should suppress it while 0.5 keeps it.
    assert "<PHONE_NUMBER>" in mask_pii(KNOWN_PII_TEXT, settings=lenient)
    assert "<PHONE_NUMBER>" not in mask_pii(KNOWN_PII_TEXT, settings=strict)


def test_pii_enabled_false_disables_masking(settings: Settings) -> None:
    disabled = settings.model_copy(update={"pii_enabled": False})

    assert mask_pii(KNOWN_PII_TEXT, settings=disabled) == KNOWN_PII_TEXT


def test_filter_documents_masks_pii_leaking_from_the_vector_store(settings: Settings) -> None:
    chunks = [
        Document(page_content=KNOWN_PII_TEXT, metadata={"doc_id": "leaked.md#contact"}),
        Document(page_content=CLEAN_TEXT, metadata={"doc_id": "rules.md#time-limits"}),
    ]

    filtered = filter_documents(chunks, settings=settings)

    assert "john.smith@example.com" not in filtered[0].page_content
    assert "<EMAIL_ADDRESS>" in filtered[0].page_content
    assert filtered[0].metadata == chunks[0].metadata
    assert filtered[1].page_content == CLEAN_TEXT
