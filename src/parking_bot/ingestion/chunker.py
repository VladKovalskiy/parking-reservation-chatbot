"""Split loaded sections into embedding-sized chunks."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document], *, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split each section Document into one or more chunks.

    Chunks keep the parent section's metadata (doc_id, source, anchor) so a
    retrieved chunk can still be matched against golden_set.jsonl entries,
    plus a chunk_index to tell multiple chunks from the same section apart.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Document] = []
    for doc in documents:
        for i, piece in enumerate(splitter.split_text(doc.page_content)):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={**doc.metadata, "chunk_index": i},
                )
            )
    return chunks
