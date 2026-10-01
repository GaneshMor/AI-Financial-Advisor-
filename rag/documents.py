"""
Load knowledge base markdown files and split them into chunks.

File format (see rag/knowledge_base/*.md):
    ---
    title: ...
    source_org: ...
    sources:
      - Document name | https://url
    last_checked: YYYY-MM-DD
    ---
    # Title
    ## Section
    text...

Chunking: one chunk per "## " section. A section longer than
config.RAG_CHUNK_CHARS is split on paragraph boundaries, and the previous
paragraph is repeated at the start of the next piece (overlap), so a sentence
is never cut in half. Each chunk starts with "Title > Section" so it still
makes sense when read alone.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import config


@dataclass
class KBDocument:
    file: str
    title: str
    source_org: str
    sources: list[tuple[str, str]]          # (name, url)
    last_checked: str
    body: str


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    title: str
    section: str
    text: str
    source_org: str
    source_url: str | None
    source_names: list[str] = field(default_factory=list)


def parse_document(path: Path) -> KBDocument:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise ValueError(f"{path.name}: missing front matter")
    _, header, body = raw.split("---", 2)
    meta: dict = {"sources": []}
    for line in header.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            name, _, url = stripped[2:].partition("|")
            meta["sources"].append((name.strip(), url.strip()))
        elif ":" in stripped:
            key, _, value = stripped.partition(":")
            if value.strip():
                meta[key.strip()] = value.strip()
    for required in ("title", "source_org"):
        if required not in meta:
            raise ValueError(f"{path.name}: front matter needs '{required}'")
    if not meta["sources"]:
        raise ValueError(f"{path.name}: front matter needs at least one source")
    return KBDocument(path.name, meta["title"], meta["source_org"], meta["sources"],
                      meta.get("last_checked", ""), body.strip())


def _split_long(paragraphs: list[str], limit: int) -> list[str]:
    pieces, current = [], []
    for para in paragraphs:
        if current and len("\n\n".join(current + [para])) > limit:
            pieces.append("\n\n".join(current))
            current = [current[-1]]  # overlap: repeat the last paragraph
        current.append(para)
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_document(doc: KBDocument, limit: int | None = None) -> list[Chunk]:
    limit = limit or config.RAG_CHUNK_CHARS
    sections: list[tuple[str, list[str]]] = []
    heading, paras = "Overview", []
    for block in doc.body.split("\n\n"):
        block = block.strip()
        if not block or block.startswith("# "):
            continue
        if block.startswith("## "):
            first, _, rest = block.partition("\n")
            if paras:
                sections.append((heading, paras))
            heading, paras = first[3:].strip(), []
            if rest.strip():
                paras.append(rest.strip())
        else:
            paras.append(block)
    if paras:
        sections.append((heading, paras))

    chunks = []
    for heading, paras in sections:
        for i, piece in enumerate(_split_long(paras, limit)):
            text = f"{doc.title} > {heading}\n{piece}"
            cid = f"{doc.file}#{heading}#{i}"
            chunks.append(Chunk(
                chunk_id=cid, source_file=doc.file, title=doc.title, section=heading, text=text,
                source_org=doc.source_org, source_url=doc.sources[0][1] if doc.sources else None,
                source_names=[n for n, _ in doc.sources],
            ))
    return chunks


def load_chunks(kb_dir: Path | None = None) -> list[Chunk]:
    kb_dir = Path(kb_dir or config.KNOWLEDGE_BASE_DIR)
    files = sorted(kb_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {kb_dir}")
    chunks: list[Chunk] = []
    for f in files:
        chunks.extend(chunk_document(parse_document(f)))
    return chunks


def kb_fingerprint(kb_dir: Path | None = None) -> str:
    """Hash of all knowledge base files; used to detect an out of date index."""
    kb_dir = Path(kb_dir or config.KNOWLEDGE_BASE_DIR)
    h = hashlib.sha256()
    for f in sorted(kb_dir.glob("*.md")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]
