from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, SparseVectorParams, Distance


COLLECTION_NAME = "college_hybrid_knowledge"

def get_qdrant_client() -> QdrantClient:
    """Opens a loacal  , file-based Qdrant storage engine ."""
    return QdrantClient(path="college_rag_storage")

def init_collection(client: QdrantClient):
    """Builds the storage collection with correct mathematical vector spaces. """

    try:
        client.get_collection(COLLECTION_NAME)
        print(f"✅ Qdrant Collection '{COLLECTION_NAME}' loaded successfully.")
    except Exception:
        print(f" Initializing a fresh hybrid collection: '{COLLECTION_NAME}'...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            
            vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
            
            sparse_vectors_config={"sparse": SparseVectorParams()}
        )
        print(" Collection initialization complete.")