# core/retriever.py
from config.database import get_qdrant_client, COLLECTION_NAME
from fastembed import TextEmbedding, SparseTextEmbedding
# Ensure FusionQuery and Fusion are included in your imports
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchValue, FusionQuery, Fusion

class CollegeKnowledgeRetriever:
    def __init__(self):
        """ Initializes the database client and embedding engines for searching. """
        self.client = get_qdrant_client()
        self.dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    def search_knowledge_base(self, user_query: str, target_category: str = None, top_k: int = 3) -> list:
        """
        Converts query to vectors, performs a hybrid search in Qdrant, and applies metadata filtering.
        """
        # Generate search vectors for the user's question
        dense_vector = list(self.dense_model.embed([user_query]))[0].tolist()
        
        sparse_vector_raw = list(self.sparse_model.embed([user_query]))[0]
        sparse_vector = {
            "indices": sparse_vector_raw.indices.tolist(),
            "values": sparse_vector_raw.values.tolist()
        }

        # Setup metadata filter if a specific category is requested
        query_filter = None
        if target_category and target_category != "All":
            query_filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=target_category))]
            )

        # UPDATED HYBRID EXECUTION LAYER
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
                Prefetch(query=sparse_vector, using="sparse", limit=top_k * 2)
            ],
            # Use FusionQuery structure from models module instead of client.Query
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=top_k
        )

        retrieved_contexts = [point.payload["text"] for point in results.points if point.payload]
        return retrieved_contexts