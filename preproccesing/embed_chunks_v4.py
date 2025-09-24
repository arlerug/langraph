# -*- coding: utf-8 -*-
from pathlib import Path
import os
import json
import numpy as np
from tqdm import tqdm
from langchain_community.embeddings import OllamaEmbeddings

# === PERCORSI ===
CHUNKS_DIR = Path(r"C:\Users\Ale\Desktop\langchain\db\data_chunks")
# usa una cartella nuova per distinguere dagli embedding Jina v4
EMB_DIR    = Path(r"C:\Users\Ale\Desktop\langchain\db\data_embeddings_mxbai")

# === MODELLO OLLAMA ===
MODEL_NAME = "mxbai-embed-large"
# prova a leggere da env (utile in docker). Fallback: localhost
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def iter_chunk_files(root: Path):
    for f in root.rglob("*.txt"):
        yield f

def main():
    if not CHUNKS_DIR.exists():
        raise SystemExit(f"Cartella chunk non trovata: {CHUNKS_DIR}")

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    manifest_in  = CHUNKS_DIR / "manifest.jsonl"
    manifest_out = EMB_DIR / "manifest_embeddings.jsonl"

    print(f"🔹 Inizializzo {MODEL_NAME} su {OLLAMA_URL} …")
    embedder = OllamaEmbeddings(model=MODEL_NAME, base_url=OLLAMA_URL)

    # Determina la dimensione embedding con una probe
    probe_vec = embedder.embed_query("probe")
    EMB_DIM = len(probe_vec)
    (EMB_DIR / "EMBEDDING_DIM.txt").write_text(str(EMB_DIM), encoding="utf-8")
    print(f"✅ Dim embedding: {EMB_DIM}")

    # mappa chunk -> metadati (se presente)
    src_by_chunk = {}
    if manifest_in.exists():
        with manifest_in.open("r", encoding="utf-8") as mf:
            for line in mf:
                try:
                    rec = json.loads(line)
                    src_by_chunk[Path(rec["chunk_path"]).resolve()] = rec
                except Exception:
                    pass

    files = list(iter_chunk_files(CHUNKS_DIR))
    if not files:
        raise SystemExit("Nessun chunk trovato. Esegui prima: python chunk_texts.py")

    total = 0
    with open(manifest_out, "w", encoding="utf-8") as mf_out:
        for f in tqdm(files, desc="Embeddings mxbai"):
            text = f.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue

            # calcola embedding con Ollama
            emb = embedder.embed_query(text)         # -> list[float]
            emb = np.asarray(emb, dtype=np.float32)  # salva come float32

            # salva npy in cartelle parallele a data_chunks
            rel = f.relative_to(CHUNKS_DIR)
            out_dir = EMB_DIR / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_npy = out_dir / (rel.stem + ".npy")
            np.save(out_npy, emb)

            info = src_by_chunk.get(f.resolve(), {})
            row = {
                "chunk_path": str(f.resolve()),
                "embedding_path": str(out_npy.resolve()),
                "embedding_dim": EMB_DIM,
                "source_path": info.get("source_path"),
                "source_name": info.get("source_name"),
                "source_dir": info.get("source_dir"),
                "chunk_index": info.get("chunk_index"),
                "chunk_size_chars": info.get("chunk_size_chars"),
                "model_name": MODEL_NAME,
                "vector_type": "single",
            }
            mf_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

    print(f"\nFATTO ✓  Chunk embeddati: {total}")
    print(f"Output: {EMB_DIR} (manifest_embeddings.jsonl, EMBEDDING_DIM.txt)")

if __name__ == "__main__":
    main()
