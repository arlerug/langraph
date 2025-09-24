# -*- coding: utf-8 -*-
from pathlib import Path
import json, uuid, time
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

# === Percorsi e settaggi ===
# cartella con i .txt già chunkati
CHUNKS_DIR = Path(r"C:\Users\Ale\Desktop\langchain\db\data_chunks")

# cartella dove salvare gli embedding
EMB_DIR = Path(r"C:\Users\Ale\Desktop\langchain\db\data_embeddings_mxbai")

# file manifest con metadati sugli embedding
MANIFEST = EMB_DIR / "manifest_embeddings.jsonl"

# file che contiene la dimensione del vettore
DIM_FILE = EMB_DIR / "EMBEDDING_DIM.txt"

QDRANT_URL = "http://localhost:6333"
COLLECTION = "kb_legale_it"
BATCH_SIZE = 256
RETRIES    = 3
RETRY_SLEEP= 1.5  # sec

DEFAULT_DOMAIN = "wesafe_cert_notarile"     # utile per filtrare lato recall

def ensure_collection(client: QdrantClient, collection: str, want_dim: int):
    """Controlla/ricrea la collection con la dimensione desiderata."""
    try:
        info = client.get_collection(collection)
        # alcune versioni espongono la size sotto config.params.vectors.size
        try:
            have_dim = info.config.params.vectors.size
        except Exception:
            # fallback grossolano: prova a usare vectors_count (non è la dim!)
            have_dim = want_dim
        if have_dim != want_dim:
            print(f"[!] Mismatch dim: collection={have_dim} vs embeddings={want_dim}. Ricreo…")
            client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=want_dim, distance=Distance.COSINE),
            )
    except UnexpectedResponse:
        print(f"[i] Collection '{collection}' non trovata. La creo…")
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=want_dim, distance=Distance.COSINE),
        )

def load_chunk_text(chunk_path_str: str, limit_chars: int = 6000) -> str:
    """Legge il testo del chunk per metterlo nel payload (utile per il recall)."""
    try:
        p = Path(chunk_path_str)
        if not p.exists():
            # manifest salvato con path assoluto? prova a risolvere relativo
            rel = Path(CHUNKS_DIR / Path(chunk_path_str).name)
            p = rel if rel.exists() else p
        txt = p.read_text(encoding="utf-8", errors="ignore")
        txt = txt.strip()
        return txt[:limit_chars]
    except Exception:
        return ""

def main():
    if not MANIFEST.exists():
        raise SystemExit(f"Manifest non trovato: {MANIFEST}")
    if not DIM_FILE.exists():
        raise SystemExit(f"File dimensione non trovato: {DIM_FILE}")

    emb_dim = int(DIM_FILE.read_text(encoding="utf-8").strip())
    if emb_dim <= 0:
        raise SystemExit(f"Dimensione embedding non valida: {emb_dim}")

    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client, COLLECTION, emb_dim)

    batch = []
    total = 0

    with MANIFEST.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Upload Qdrant"):
            rec = json.loads(line)
            emb_path = Path(rec["embedding_path"])
            if not emb_path.exists():
                continue

            vec = np.load(emb_path)
            if vec.shape[-1] != emb_dim:
                print(f"[!] Skip: dim vettore {vec.shape[-1]} != attesa {emb_dim}  → {emb_path}")
                continue

            vector = vec.astype(np.float32).tolist()

            # testo del chunk per il recall
            chunk_path = rec.get("chunk_path") or ""
            chunk_text = load_chunk_text(chunk_path)

            payload = {
                "text": chunk_text,                    # <— importante per il recall
                "chunk_path": chunk_path,
                "source_path": rec.get("source_path"),
                "source_name": rec.get("source_name"),
                "source_dir":  rec.get("source_dir"),
                "chunk_index": rec.get("chunk_index"),
                "chunk_size_chars": rec.get("chunk_size_chars"),
                "model_name":  rec.get("model_name"),
                "vector_type": rec.get("vector_type"),
                "domain": DEFAULT_DOMAIN,              # per filtri lato Cheshire/Qdrant
            }

            batch.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            ))

            if len(batch) >= BATCH_SIZE:
                for attempt in range(RETRIES):
                    try:
                        client.upsert(collection_name=COLLECTION, points=batch)
                        total += len(batch)
                        batch = []
                        break
                    except Exception as e:
                        if attempt == RETRIES - 1:
                            raise
                        print(f"[warn] upsert batch failed ({e}), retry in {RETRY_SLEEP}s…")
                        time.sleep(RETRY_SLEEP)

    if batch:
        client.upsert(collection_name=COLLECTION, points=batch)
        total += len(batch)

    print(f"\n✅ Upload completato. Chunk inseriti: {total}")
    print(f"Collection: {COLLECTION} @ {QDRANT_URL} | dim={emb_dim}")

if __name__ == "__main__":
    main()
