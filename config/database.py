import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, SparseVectorParams, Distance

COLLECTION_NAME = "college_hybrid_knowledge"

def get_qdrant_client() -> QdrantClient:
    """Opens a local, file-based Qdrant storage engine with a robust relative path."""
    # Get the absolute path of the current directory (config/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Point to the storage folder one level up in the project root
    storage_path = os.path.join(current_dir, "..", "college_rag_storage")
    
    return QdrantClient(path=storage_path)

def init_collection(client: QdrantClient):
    """Builds the storage collection with correct mathematical vector spaces."""
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"✅ Qdrant Collection '{COLLECTION_NAME}' loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load existing collection: {e}")