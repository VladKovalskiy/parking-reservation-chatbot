from langchain_core.documents import Document
from langchain_milvus import Milvus

from parking_bot.config import Settings
from parking_bot.eval.harness import GoldenExample, load_golden_set, run_eval
from parking_bot.llm.embeddings import build_embeddings

DOCS = [
    Document(page_content="Parking costs $5 per hour.", metadata={"doc_id": "pricing"}),
    Document(page_content="The garage is on Main Street.", metadata={"doc_id": "location"}),
    Document(page_content="Reservations need confirmation.", metadata={"doc_id": "booking"}),
]


def _seeded_store(settings: Settings) -> Milvus:
    embeddings = build_embeddings(settings)
    return Milvus.from_documents(
        DOCS,
        embeddings,
        collection_name=settings.milvus_collection,
        connection_args={"uri": settings.milvus_connection_uri},
        drop_old=True,
    )


def test_run_eval_scores_a_perfect_match_as_one_across_all_ks(settings: Settings) -> None:
    store = _seeded_store(settings)
    golden_set = [GoldenExample(question=DOCS[0].page_content, relevant_doc_ids=["pricing"])]

    report = run_eval(golden_set, ks=[1, 3], store=store, settings=settings)

    assert report["num_questions"] == 1
    assert report["metrics"]["1"]["recall_at_k"] == 1.0
    assert report["metrics"]["1"]["precision_at_k"] == 1.0
    assert report["metrics"]["3"]["recall_at_k"] == 1.0


def test_run_eval_averages_across_multiple_questions(settings: Settings) -> None:
    store = _seeded_store(settings)
    golden_set = [
        GoldenExample(question=DOCS[0].page_content, relevant_doc_ids=["pricing"]),
        GoldenExample(question="something with no matching doc", relevant_doc_ids=["missing"]),
    ]

    report = run_eval(golden_set, ks=[1], store=store, settings=settings)

    # one perfect hit (recall 1.0) and one total miss (recall 0.0) -> mean 0.5
    assert report["metrics"]["1"]["recall_at_k"] == 0.5


def test_load_golden_set_parses_question_and_relevant_doc_ids(tmp_path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"question": "Where is it?", "relevant_doc_ids": ["general.md#overview"], '
        '"expected_answer_contains": []}\n',
        encoding="utf-8",
    )

    examples = load_golden_set(path)

    assert examples == [
        GoldenExample(question="Where is it?", relevant_doc_ids=["general.md#overview"])
    ]
