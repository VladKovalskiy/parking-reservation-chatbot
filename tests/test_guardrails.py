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
