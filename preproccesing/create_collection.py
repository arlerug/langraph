# create_collection_e5.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")

client.recreate_collection(
    collection_name="chat_memory",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

print("✅ Collection 'kb_legale_it' creata (1024-dim, COSINE, e5-large)")
