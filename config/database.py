import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, SparseVectorParams, Distance

COLLECTION_NAME = "college_hybrid_knowledge"

def get_qdrant_client() -> QdrantClient:
    """Safely connects to Qdrant Cloud using Streamlit secrets or environment variables (.env)."""
    qdrant_url = None
    qdrant_api_key = None

    # 1. Try reading from Streamlit secrets (for cloud deployment)
    try:
        import streamlit as st
        qdrant_url = st.secrets.get("QDRANT_URL")
        qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
    except Exception:
        pass  # Not running in a Streamlit context

    # 2. Fallback to environment variables / .env file (for local scripts)
    if not qdrant_url:
        qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_api_key:
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError("❌ QDRANT_URL and QDRANT_API_KEY must be set in Streamlit secrets or your .env file.")

    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

def init_collection(client: QdrantClient):
    """Ensures the hybrid collection exists on Qdrant Cloud and has required payload indexes."""
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"✅ Qdrant Collection '{COLLECTION_NAME}' loaded successfully.")
    except Exception:
        print(f"⚠️ Initializing fresh hybrid collection '{COLLECTION_NAME}'...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()}
        )
    
    # 🔑 CRITICAL FIX: Create payload index for 'category' to satisfy Qdrant Strict Mode
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="category",
            field_schema="keyword"
        )
        print("✅ Payload index created successfully for field: 'category'")
    except Exception as e:
        print(f"ℹ️ Payload index status: {e}")