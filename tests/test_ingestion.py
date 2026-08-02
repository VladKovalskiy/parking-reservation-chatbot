from pathlib import Path

from langchain_core.documents import Document

from parking_bot.config import Settings
from parking_bot.ingestion.chunker import chunk_documents
from parking_bot.ingestion.loader import load_static_documents
from parking_bot.ingestion.pipeline import run_ingestion


def test_chunker_splits_long_sections_respecting_chunk_size_and_overlap() -> None:
    long_text = " ".join(f"word{i}" for i in range(200))
    doc = Document(page_content=long_text, metadata={"doc_id": "x.md#y"})

    chunks = chunk_documents([doc], chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 50 for c in chunks)
    # overlap: the last word of one chunk reappears at the start of the next
    last_word = chunks[0].page_content.split()[-1]
    assert chunks[1].page_content.startswith(last_word)


def test_chunk_metadata_carries_doc_id_source_anchor_and_chunk_index(tmp_path: Path) -> None:
    (tmp_path / "sample.md").write_text(
        '# Sample\n\n<a id="one"></a>\n## One\n\n' + ("filler word " * 50) + "\n",
        encoding="utf-8",
    )

    sections = load_static_documents(tmp_path)
    chunks = chunk_documents(sections, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["doc_id"] == "sample.md#one"
        assert chunk.metadata["source"] == "sample.md"
        assert chunk.metadata["anchor"] == "one"
        assert chunk.metadata["chunk_index"] == i


def test_loader_parses_every_anchor_in_the_real_static_docs() -> None:
    static_dir = Path(__file__).resolve().parents[1] / "data" / "static"

    docs = load_static_documents(static_dir)
    doc_ids = {d.metadata["doc_id"] for d in docs}

    assert "general.md#overview" in doc_ids
    assert "booking.md#how-to-reserve" in doc_ids
    assert len(docs) == 20  # 4 static files x 5 anchored sections each


def test_run_ingestion_is_idempotent(settings: Settings, tmp_path: Path) -> None:
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "a.md").write_text(
        '# A\n\n<a id="one"></a>\n## One\n\nSome short parking content.\n',
        encoding="utf-8",
    )

    first_run_chunks = run_ingestion(settings, static_dir=static_dir)
    second_run_chunks = run_ingestion(settings, static_dir=static_dir)

    assert first_run_chunks == second_run_chunks == 1
