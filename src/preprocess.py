"""
src/preprocess.py
Corpus builder (Weeks 1-2, corrected in the Week 5/6 audit).

Pipeline:  raw files -> text (pdfplumber | OCR fallback | python-docx) -> cleaned text
           -> sentence-window chunks -> Data/processed/corpus.json

Fixes compared with the original version
----------------------------------------
* Scanned PDFs (Placement Policy, Internship Policy, Academic Calendars, IPT
  guidelines, ...) have NO text layer. pdfplumber silently returned "" for them,
  so the most important policy documents were never indexed. Pages without a text
  layer are now OCR-ed (RapidOCR / ONNX, runs on CPU) and cached on disk.
* Policy documents delivered only as .docx (Department objectives, FAQ, Important
  Instructions, ...) are now ingested too.
* Every chunk carries `doc_path` ("Folder/file.pdf"), `category`, `page` and
  `extraction` so citations can be reported as "folder/file.pdf" and filenames that
  occur in several folders (OE 4 / OE 5) are no longer ambiguous.
* Course study material (calculus / physics textbooks, tutorials, lecture slides)
  is NOT policy and made up 70 % of the old index (Thomas_Calculus.pdf alone was 46 %
  of all chunks). It is excluded by default (`--include-study-material` restores it).
* Superseded editions of the Student Resource Book (SRB 2016 ... SRB 2025) are
  excluded by default so that out-of-date rules cannot be cited (`--include-superseded`).
* Chunk size is capped (the old corpus had chunks of up to 30,000 characters).

Usage:
    python src/preprocess.py                      # build corpus.json (+ OCR cache)
    python src/preprocess.py --dry-run            # list what would be ingested
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import inspect

if not hasattr(inspect, "formatargspec"):  # nltk compat shim for Python 3.14
    inspect.formatargspec = lambda *a, **k: ""  # type: ignore[attr-defined]

import nltk
from nltk.tokenize import sent_tokenize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("preprocess")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "Data" / "Raw" / "STUDENT HANDOUT DOCUMENTS"
DEFAULT_OUT_DIR = PROJECT_ROOT / "Data" / "processed"

WINDOW_SENTENCES = 6
OVERLAP_SENTENCES = 2
MAX_CHUNK_CHARS = 1500
MAX_SENTENCE_CHARS = 500
OCR_MIN_CHARS = 25  # a page with fewer extracted characters is treated as "scanned"


# ───────────────────────────── helpers ─────────────────────────────

def setup_nltk() -> None:
    for resource, path in (("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
                           ("wordnet", "corpora/wordnet.zip")):
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception as exc:  # offline machine: tokenizer may already exist
                log.warning("Could not download NLTK resource %s: %s", resource, exc)


def clean_text(text: str) -> str:
    """Collapse whitespace; strip stray control characters and bare page numbers."""
    if not text:
        return ""
    text = text.replace(" ", " ").replace(" ", " ").replace("﻿", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_long_sentence(sentence: str, limit: int = MAX_SENTENCE_CHARS) -> List[str]:
    if len(sentence) <= limit:
        return [sentence]
    parts, current = [], []
    size = 0
    for word in sentence.split(" "):
        if size + len(word) + 1 > limit and current:
            parts.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += len(word) + 1
    if current:
        parts.append(" ".join(current))
    return parts


def chunk_sentences(
    sentences: List[Tuple[str, int]],
    window: int = WINDOW_SENTENCES,
    overlap: int = OVERLAP_SENTENCES,
    max_chars: int = MAX_CHUNK_CHARS,
) -> List[Tuple[str, int]]:
    """Sentence-window chunking. `sentences` is a list of (sentence, page). Returns (text, first_page)."""
    chunks: List[Tuple[str, int]] = []
    n = len(sentences)
    if n == 0:
        return chunks
    step = max(1, window - overlap)
    i = 0
    while i < n:
        text_parts: List[str] = []
        size = 0
        j = i
        while j < n and (j - i) < window:
            s = sentences[j][0]
            if text_parts and size + len(s) + 1 > max_chars:
                break
            text_parts.append(s)
            size += len(s) + 1
            j += 1
        chunks.append((" ".join(text_parts), sentences[i][1]))
        if j >= n:
            break
        i += max(1, (j - i) - overlap)
    return chunks


def chunk_text(text: str, window_sentences: int = WINDOW_SENTENCES,
               overlap_sentences: int = OVERLAP_SENTENCES) -> List[str]:
    """Backward-compatible helper used by older scripts."""
    sents = [(p, 1) for s in sent_tokenize(text) for p in _split_long_sentence(s)]
    return [c for c, _ in chunk_sentences(sents, window_sentences, overlap_sentences)]


# ───────────────────────────── extraction ─────────────────────────────

_OCR_ENGINE = None


def _get_ocr():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR()
        except Exception as exc:
            log.error("OCR engine unavailable (%s). Install `rapidocr-onnxruntime` to ingest scanned PDFs.", exc)
            _OCR_ENGINE = False
    return _OCR_ENGINE or None


def _render_page_png(pdf_path: Path, page_index: int, dpi: int = 220) -> Optional[bytes]:
    try:
        import pymupdf  # type: ignore
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore
        except ImportError:
            pymupdf = None
    if pymupdf is not None:
        with pymupdf.open(str(pdf_path)) as doc:
            return doc[page_index].get_pixmap(dpi=dpi).tobytes("png")
    try:
        import pypdfium2 as pdfium
        import io
        pdf = pdfium.PdfDocument(str(pdf_path))
        img = pdf[page_index].render(scale=dpi / 72).to_pil()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        log.error("Cannot render %s p%d: %s", pdf_path.name, page_index + 1, exc)
        return None


def ocr_page(pdf_path: Path, page_index: int, cache_dir: Path) -> str:
    """OCR one PDF page. Result is cached on disk so rebuilding the corpus is cheap."""
    key = hashlib.sha1(f"{pdf_path.as_posix()}::{pdf_path.stat().st_size}::{page_index}".encode()).hexdigest()
    cache_file = cache_dir / f"{key}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    engine = _get_ocr()
    if engine is None:
        return ""
    png = _render_page_png(pdf_path, page_index)
    if png is None:
        return ""
    result, _ = engine(png)
    lines = []
    for item in result or []:
        # item = [box, text, confidence]
        if len(item) >= 3 and float(item[2]) < 0.30:
            continue
        lines.append(item[1])
    text = "\n".join(lines)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text, encoding="utf-8")
    return text


def extract_pdf_pages(pdf_path: Path, cache_dir: Path, use_ocr: bool = True) -> Iterator[Tuple[int, str, str]]:
    """Yield (page_number, text, method) for each page. method is 'text' or 'ocr'."""
    import pdfplumber
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            text = ""
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                log.warning("Error reading %s p%d: %s", pdf_path.name, idx + 1, exc)
            if len(text.strip()) >= OCR_MIN_CHARS or not use_ocr:
                yield idx + 1, text, "text"
                continue
            if page.images:  # blank text layer but picture on page => scanned
                yield idx + 1, ocr_page(pdf_path, idx, cache_dir), "ocr"
            else:
                yield idx + 1, text, "text"


def extract_docx(path: Path) -> str:
    import docx  # python-docx
    document = docx.Document(str(path))
    parts: List[str] = []
    for element in document.element.body.iterchildren():
        if element.tag.endswith("}p"):
            t = "".join(x.text or "" for x in element.iter() if x.tag.endswith("}t"))
            if t.strip():
                parts.append(t.strip() + ".")
        elif element.tag.endswith("}tbl"):
            import docx.table
            table = docx.table.Table(element, document)
            for row in table.rows:
                cells = []
                for c in row.cells:
                    if c.text.strip() and c.text.strip() not in cells:
                        cells.append(c.text.strip().replace("\n", "; "))
                if cells:
                    parts.append(" | ".join(cells) + ".")
    return "\n".join(parts)


# ───────────────────────────── scope rules ─────────────────────────────

def is_study_material(rel: str) -> bool:
    return "study material" in rel.lower()


def srb_edition(rel: str) -> Optional[int]:
    m = re.search(r"SRB\s*(\d{4})", Path(rel).name, flags=re.I)
    return int(m.group(1)) if m else None


def collect_files(raw_dir: Path, include_study_material: bool, include_superseded: bool
                  ) -> Tuple[List[Path], Dict[str, str]]:
    """Return files to ingest and a {rel_path: reason} dict for everything skipped."""
    all_files = [p for p in sorted(raw_dir.rglob("*")) if p.is_file()]
    pdf_stems = {p.stem.lower() for p in all_files if p.suffix.lower() == ".pdf"}
    srb_years = [y for y in (srb_edition(p.relative_to(raw_dir).as_posix()) for p in all_files
                             if p.suffix.lower() == ".pdf") if y]
    latest_srb = max(srb_years) if srb_years else None

    keep, skipped = [], {}
    for p in all_files:
        rel = p.relative_to(raw_dir).as_posix()
        suffix = p.suffix.lower()
        if suffix not in (".pdf", ".docx"):
            continue  # video / image / pptx are not text policy sources
        if not include_study_material and is_study_material(rel):
            skipped[rel] = "study material (not policy)"
            continue
        if suffix == ".docx" and p.stem.lower() in pdf_stems:
            skipped[rel] = "duplicate of a PDF"
            continue
        ed = srb_edition(rel)
        if not include_superseded and ed is not None and latest_srb and ed < latest_srb:
            skipped[rel] = f"superseded Student Resource Book edition (latest is {latest_srb})"
            continue
        keep.append(p)
    return keep, skipped


# ───────────────────────────── main build ─────────────────────────────

def build_corpus(raw_dir: Path = DEFAULT_RAW_DIR, out_dir: Path = DEFAULT_OUT_DIR,
                 include_study_material: bool = False, include_superseded: bool = False,
                 use_ocr: bool = True, dry_run: bool = False) -> List[dict]:
    setup_nltk()
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "ocr_cache"

    files, skipped = collect_files(raw_dir, include_study_material, include_superseded)
    log.info("Ingesting %d files (skipped %d).", len(files), len(skipped))
    if dry_run:
        for f in files:
            print("KEEP ", f.relative_to(raw_dir).as_posix())
        for rel, why in skipped.items():
            print("SKIP ", rel, "->", why)
        return []

    corpus: List[dict] = []
    doc_report: List[dict] = []
    chunk_id = 0
    for path in files:
        rel = path.relative_to(raw_dir).as_posix()
        category = rel.split("/")[0]
        sentences: List[Tuple[str, int]] = []
        methods = set()
        try:
            if path.suffix.lower() == ".pdf":
                pages = extract_pdf_pages(path, cache_dir, use_ocr=use_ocr)
            else:
                pages = iter([(1, extract_docx(path), "docx")])
            for page_no, text, method in pages:
                text = clean_text(text) if method != "ocr" else clean_text(text.replace("\n", " "))
                if not text:
                    continue
                methods.add(method)
                for s in sent_tokenize(text):
                    for piece in _split_long_sentence(s.strip()):
                        if piece:
                            sentences.append((piece, page_no))
        except Exception as exc:
            log.error("Failed on %s: %s", rel, exc)
            doc_report.append({"doc_path": rel, "status": f"error: {exc}", "chunks": 0})
            continue

        chunks = chunk_sentences(sentences)
        for text, page in chunks:
            corpus.append({
                "chunk_id": chunk_id,
                "source_document": path.name,
                "doc_path": rel,
                "category": category,
                "page": page,
                "extraction": "ocr" if "ocr" in methods else ("docx" if "docx" in methods else "text"),
                "text": text,
            })
            chunk_id += 1
        status = "ok" if chunks else "EMPTY (no extractable text)"
        doc_report.append({"doc_path": rel, "status": status, "chunks": len(chunks),
                           "methods": sorted(methods)})
        log.info("%-95s %4d chunks %s", rel[:95], len(chunks), sorted(methods))

    (out_dir / "corpus.json").write_text(json.dumps(corpus, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "corpus_manifest.json").write_text(json.dumps(
        {"n_chunks": len(corpus), "n_documents": len(doc_report), "documents": doc_report,
         "skipped": skipped}, indent=1, ensure_ascii=False), encoding="utf-8")
    empties = [d["doc_path"] for d in doc_report if d["chunks"] == 0]
    log.info("Wrote %d chunks from %d documents to %s", len(corpus), len(doc_report), out_dir / "corpus.json")
    if empties:
        log.warning("%d documents produced no text: %s", len(empties), empties)
    return corpus


def process_pdfs(raw_dir=DEFAULT_RAW_DIR, processed_dir=DEFAULT_OUT_DIR):
    """Backward-compatible entry point used by the Week 1-2 scripts."""
    return build_corpus(Path(raw_dir), Path(processed_dir))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the university policy corpus")
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--include-study-material", action="store_true")
    ap.add_argument("--include-superseded", action="store_true")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    build_corpus(Path(a.raw_dir), Path(a.out_dir), a.include_study_material, a.include_superseded,
                 use_ocr=not a.no_ocr, dry_run=a.dry_run)
