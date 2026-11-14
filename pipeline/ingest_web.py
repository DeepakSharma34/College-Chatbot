# pipeline/ingest_web.py
import uuid
import os
import sys

# 1. Load environment variables from .env file first
from dotenv import load_dotenv
load_dotenv()

# 2. Fix path to resolve root config module if running from pipeline/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.database import get_qdrant_client, init_collection, COLLECTION_NAME
from core.scraper import scrape_url_text
from core.chunker import split_text_into_chunks
from models.schemas import collegeDataPayload
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client.models import PointStruct

def get_manual_category(url: str) -> str:
    """
    Checks the URL string directly and routes it to the correct Pydantic domain category.
    Returns None if no keyword matches.
    """
    url_lower = url.lower()
    if "placement" in url_lower or "training" in url_lower:
        return "Placements"
    elif "hostel" in url_lower or "facility" in url_lower:
        return "Hostel"
    elif "syllabus" in url_lower or "curriculum" in url_lower:
        return "Syllabus"
    elif "courses" in url_lower or "admission" in url_lower:
        return "Courses"
    elif "academics" in url_lower or "department" in url_lower:
        return "Academics"
    elif "research" in url_lower or "development" in url_lower:
        return "Academics" 
    else:
        return None 

def classify_chunk_by_keywords(text: str, url: str) -> str:
    """Classifies a chunk instantly using keywords without consuming Gemini API quota."""
    text_lower = text.lower()
    url_lower = url.lower()

    if "placement" in url_lower or "training" in url_lower or "recruiter" in text_lower:
        return "Placements"
    elif "hostel" in url_lower or "facility" in url_lower or "room" in text_lower or "mess" in text_lower:
        return "Hostel"
    elif "fee" in text_lower or "admission" in text_lower or "eligibility" in text_lower or "tuition" in text_lower:
        return "Admissions_Fees"
    elif "syllabus" in url_lower or "curriculum" in url_lower or "semester" in text_lower:
        return "Syllabus"
    elif "course" in text_lower or "b.tech" in text_lower or "mca" in text_lower or "bca" in text_lower:
        return "Academics_Courses"
    elif "research" in url_lower or "development" in url_lower or "lab" in text_lower:
        return "Academics"
    else:
        return "Academics_Courses"

def run_web_ingestion(urls: list):
    """
    Executes the ingestion pipeline with local keyword routing (zero Gemini API quota usage).
    """
    client = get_qdrant_client()
    init_collection(client)
    
    print("🚀 Loading Dual Embedding Engines into Memory...")
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    
    for url in urls:
        print(f"\n🔗 [WORKFLOW 1] Extracting data from: {url}")
        
        manual_category = get_manual_category(url)
        if manual_category:
            print(f"📌 URL Keyword Match: Routing all chunks explicitly to [{manual_category}]")
        
        raw_text = scrape_url_text(url)
        if not raw_text:
            print(f"⚠️ Extraction returned nothing for {url}. Skipping.")
            continue
            
        chunks = split_text_into_chunks(raw_text, chunk_size=150, overlap=30)
        print(f"📄 Text cleanly split into {len(chunks)} paragraph units.")
        
        points_to_store = []
        
        for index, chunk in enumerate(chunks):
            try:
                category = manual_category if manual_category else classify_chunk_by_keywords(chunk, url)
                
                structured_payload = collegeDataPayload(
                    text=chunk,
                    category=category,
                    source=url,
                    tags=[category.lower(), "glbitm-scraped"]
                )
                
                dense_vector = list(dense_model.embed([structured_payload.text]))[0].tolist()
                
                sparse_vector_raw = list(sparse_model.embed([structured_payload.text]))[0]
                sparse_vector = {
                    "indices": sparse_vector_raw.indices.tolist(), 
                    "values": sparse_vector_raw.values.tolist()
                }
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"dense": dense_vector, "sparse": sparse_vector},
                    payload=structured_payload.model_dump()
                )
                points_to_store.append(point)
                
            except Exception as e:
                print(f"    Validation skipped an invalid block: {e}")
                
        if points_to_store:
            print(f"💾 Writing {len(points_to_store)} nodes to Qdrant Cloud storage...")
            client.upsert(collection_name=COLLECTION_NAME, points=points_to_store)
            print(f"🏁 Done with URL!")

if __name__ == "__main__":
    college_urls = [
        "https://www.glbitm.org/training-and-placements/",
        "https://www.glbitm.org/hostel-facilities/",
        "https://www.glbitm.org/courses-and-admission/",
        "https://www.glbitm.org/academics%20-departments/",
        "https://www.glbitm.org/best-engineering-institute-in-greater-noida",
        "https://www.glbitm.org/research-development/research-development/64/241/",
    ]
    run_web_ingestion(college_urls)