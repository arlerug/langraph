# -*- coding: utf-8 -*-
"""
Chunker semplice:
- Scansiona cartelle di input per trovare .txt
- Divide ogni file in chunk (size, overlap configurabili)
- Salva i chunk come .txt in data_chunks/<sottocartella_input>/
- Scrive anche un manifest.jsonl con metadata utili

Esegui:
    python chunk_texts.py
"""

from pathlib import Path
import json

# ====== CONFIG ======
INPUT_DIRS = [
    "C:\\Users\\Ale\\Desktop\\langchain\\db\\kb_rag_wikipedia"
]


OUTPUT_DIR = Path("data_chunks")
CHUNK_SIZE = 900        # ~150–200 token
CHUNK_OVERLAP = 150     # continuità tra spezzoni
# ====================


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Divide il testo in blocchi con sovrapposizione."""
    if not text:
        return []
    chunks = []
    i, n = 0, len(text)
    while i < n:
        j = min(n, i + size)
        c = text[i:j].strip()
        if c:
            chunks.append(c)
        if j >= n:
            break
        i = j - overlap if (j - overlap) > i else j
    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "manifest.jsonl"
    total_files = 0
    total_chunks = 0

    with manifest_path.open("w", encoding="utf-8") as mf:
        for in_dir in INPUT_DIRS:
            d = Path(in_dir)
            if not d.exists():
                print(f"[skip] Cartella non trovata: {d}")
                continue

            # I chunk andranno in una sottocartella che ricalca il nome di quella di input
            out_subdir = OUTPUT_DIR / d.name
            out_subdir.mkdir(parents=True, exist_ok=True)

            for f in d.rglob("*.txt"):
                total_files += 1
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = f.read_text(errors="ignore")

                parts = (
                    chunk_text(text)
                    if len(text) > CHUNK_SIZE
                    else ([text.strip()] if text.strip() else [])
                )
                if not parts:
                    print(f"[empty] {f}")
                    continue

                # salva i chunk come .txt numerati
                stem = f.stem  # nome base senza .txt
                for idx, ch in enumerate(parts, start=1):
                    chunk_name = f"{stem}__chunk{idx:03d}.txt"
                    out_fp = out_subdir / chunk_name
                    out_fp.write_text(ch, encoding="utf-8")
                    total_chunks += 1

                    # riga nel manifest
                    row = {
                        "source_path": str(f.resolve()),
                        "source_dir": str(d.name),
                        "source_name": f.name,
                        "chunk_index": idx,
                        "chunk_path": str(out_fp.resolve()),
                        "chunk_size_chars": len(ch),
                    }
                    mf.write(json.dumps(row, ensure_ascii=False) + "\n")

                print(f"[ok] {f} -> {len(parts)} chunk")

    print(f"\nFATTO ✓  File processati: {total_files}  |  Chunk salvati: {total_chunks}")
    print(f"Output: {OUTPUT_DIR}  (con manifest.jsonl)")


if __name__ == "__main__":
    main()
