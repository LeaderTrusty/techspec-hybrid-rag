from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pymupdf4llm
from pydantic import BaseModel


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_name: str
    page: int
    section: str
    is_table: bool
    text: str


HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and len(stripped) >= 3


def _split_into_blocks(md_text: str, current_section: str) -> tuple[list[dict], str]:
    lines = md_text.splitlines()
    blocks: list[dict] = []
    prose_buf: list[str] = []
    table_buf: list[str] = []

    def flush_prose() -> None:
        text = "\n".join(prose_buf).strip()
        if text:
            blocks.append({"type": "prose", "section": current_section, "text": text})
        prose_buf.clear()

    def flush_table() -> None:
        text = "\n".join(table_buf).strip()
        if text:
            blocks.append({"type": "table", "section": current_section, "text": text})
        table_buf.clear()

    for line in lines:
        heading_match = HEADING_RE.match(line.strip())
        if heading_match:
            flush_table()
            flush_prose()
            current_section = heading_match.group(2).strip("*_ ")
            prose_buf.append(line.strip())
            continue

        if _is_table_line(line):
            flush_prose()
            table_buf.append(line.strip())
        else:
            flush_table()
            prose_buf.append(line)

    flush_table()
    flush_prose()
    return blocks, current_section


def _split_large_table(table_md: str, max_chars: int) -> list[str]:
    if len(table_md) <= max_chars:
        return [table_md]

    rows = [r for r in table_md.splitlines() if r.strip()]
    if len(rows) <= 3:
        return [table_md]

    has_header = len(rows) >= 2 and set(rows[1].replace("|", "").replace(":", "").strip()) <= {"-", " "}
    header_rows = rows[:2] if has_header else []
    body_rows = rows[2:] if has_header else rows
    header_prefix = "\n".join(header_rows)

    sub_tables: list[str] = []
    current_rows: list[str] = []
    current_len = len(header_prefix)

    for row in body_rows:
        if current_rows and (current_len + len(row) + 1) > max_chars:
            chunk_rows = (header_rows + current_rows) if has_header else current_rows
            sub_tables.append("\n".join(chunk_rows))
            current_rows = [row]
            current_len = len(header_prefix) + len(row) + 1
        else:
            current_rows.append(row)
            current_len += len(row) + 1

    if current_rows:
        chunk_rows = (header_rows + current_rows) if has_header else current_rows
        sub_tables.append("\n".join(chunk_rows))

    return sub_tables


def _split_prose(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        if len(para) > max_chars:
            if buf:
                chunks.append(buf.strip())
                buf = ""
            start = 0
            while start < len(para):
                end = min(start + max_chars, len(para))
                chunks.append(para[start:end].strip())
                if end == len(para):
                    break
                start = max(0, end - overlap_chars)
            continue

        if buf and len(buf) + len(para) + 2 > max_chars:
            chunks.append(buf.strip())
            tail = buf[-overlap_chars:].strip() if overlap_chars > 0 else ""
            buf = f"{tail}\n\n{para}".strip()
        else:
            buf = f"{buf}\n\n{para}".strip() if buf else para

    if buf:
        chunks.append(buf.strip())

    return chunks


def parse_and_chunk_pdf(
    pdf_path: Path,
    max_chars: int = 1200,
    overlap_chars: int = 180,
) -> List[DocumentChunk]:
    page_dicts = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    doc_name = pdf_path.name
    all_chunks: list[DocumentChunk] = []
    current_section = "General Overview"

    for page_idx, page_data in enumerate(page_dicts, start=1):
        raw_md = (page_data.get("text") or "").strip()
        if not raw_md:
            continue

        blocks, current_section = _split_into_blocks(raw_md, current_section)
        local_idx = 0

        for blk in blocks:
            section = blk["section"]
            if blk["type"] == "table":
                pieces = _split_large_table(blk["text"], max_chars=max_chars)
                is_table = True
            else:
                pieces = _split_prose(blk["text"], max_chars=max_chars, overlap_chars=overlap_chars)
                is_table = False

            for piece in pieces:
                if len(piece.strip()) < 30:
                    continue
                local_idx += 1
                enriched_text = f"[Doc: {doc_name} | Page {page_idx} | Section: {section}]\n{piece}"
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=f"{pdf_path.stem}_p{page_idx:03d}_c{local_idx:02d}",
                        doc_name=doc_name,
                        page=page_idx,
                        section=section,
                        is_table=is_table,
                        text=enriched_text,
                    )
                )

    return all_chunks


def ingest_directory(
    pdf_dir: Path = Path("data/datasheets"),
    out_jsonl: Path = Path("storage/chunks.jsonl"),
) -> List[DocumentChunk]:
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir.resolve()}")

    corpus: list[DocumentChunk] = []
    for pdf in pdf_files:
        doc_chunks = parse_and_chunk_pdf(pdf)
        table_count = sum(1 for c in doc_chunks if c.is_table)
        print(f"Parsed {pdf.name}: {len(doc_chunks)} chunks ({table_count} table chunks)")
        corpus.extend(doc_chunks)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for chunk in corpus:
            f.write(chunk.model_dump_json() + "\n")

    print(f"Saved {len(corpus)} total chunks to {out_jsonl}")
    return corpus


def load_chunks(jsonl_path: Path = Path("storage/chunks.jsonl")) -> List[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(DocumentChunk.model_validate_json(line))
    return chunks
