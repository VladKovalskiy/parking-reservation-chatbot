"""Load static markdown documents into anchor-scoped sections.

Each `data/static/*.md` file is split on its `<a id="...">` anchors, one
Document per section, so a section's `doc_id` metadata (e.g.
"general.md#overview") matches the identifiers already used in
`data/eval/golden_set.jsonl` — see the anchors in data/static/*.md.
"""

import re
from pathlib import Path

from langchain_core.documents import Document

_SECTION_RE = re.compile(
    r'<a id="(?P<anchor>[\w-]+)"></a>\s*\n#+\s*(?P<title>[^\n]+)\n+(?P<body>.*?)(?=\n<a id="|\Z)',
    re.DOTALL,
)


def load_static_documents(static_dir: Path) -> list[Document]:
    """Read every .md file in static_dir into one Document per anchored section."""
    documents: list[Document] = []
    for path in sorted(static_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for match in _SECTION_RE.finditer(text):
            anchor = match.group("anchor")
            title = match.group("title").strip()
            body = match.group("body").strip()
            documents.append(
                Document(
                    page_content=f"{title}\n\n{body}",
                    metadata={
                        "source": path.name,
                        "anchor": anchor,
                        "doc_id": f"{path.name}#{anchor}",
                        "title": title,
                    },
                )
            )
    return documents
