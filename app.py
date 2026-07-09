# app.py
import os
import streamlit as st
from google import genai
from dotenv import load_dotenv
from core.retriever import CollegeKnowledgeRetriever

# Load environment variables securely from your local .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY not found in your .env file. Please add it to continue.")
    st.stop()

# Initialize the modern, unified Google GenAI Client with caching
@st.cache_resource
def get_gemini_client(key: str):
    return genai.Client(api_key=key)

client = get_gemini_client(api_key)

# Page Layout Configuration Setup
st.set_page_config(page_title="College Assistant", layout="wide")
st.title("🎓 Intelligent College RAG Helpdesk")
st.markdown("---")

# Initialize and cache our hybrid vector search engine module into application memory
@st.cache_resource
def load_search_engine():
    return CollegeKnowledgeRetriever()

retriever = load_search_engine()

# Left Sidebar UI: Context Filtering Control Center
with st.sidebar:
    st.header("⚙️ Search Controls")
    st.markdown("Use these settings to focus your assistant's knowledge search area.")
    
    # Category dropdown matching our Pydantic validation domains exactly
    selected_domain = st.selectbox(
        "Focus Area Category:",
        ["All", "Placements", "Hostel", "Fees", "Syllabus", "Courses", "Academics"]
    )
    
    st.info("💡 **Tip:** Selecting a category forces Qdrant to apply a hardware-level metadata filter, keeping context processing accurate.")

# Main Layout Workspace: User Query Submission & Generation Section
st.subheader("💬 Ask Your College Queries")
user_input = st.text_input("Type your question here (e.g., 'What are the placement highlights for GL Bajaj?'):")

if user_input:
    with st.spinner("🔍 Scanning Hybrid Knowledge Base & Formulating Answer..."):
        # Step 1: Query local Qdrant database to retrieve relevant documentation blocks
        context_chunks = retriever.search_knowledge_base(
            user_query=user_input, 
            target_category=selected_domain, 
            top_k=3
        )
        
        if not context_chunks:
            st.warning("⚠️ No matching context documents found within your database for this category.")
            combined_context = "No direct documentation context found."
        else:
            # Combine individual string chunks into a single reference context canvas
            combined_context = "\n\n".join([f"[Doc {i+1}]: {text}" for i, text in enumerate(context_chunks)])
            
            # Show verified ground-truth documentation snippets inside an expander tab
            with st.expander("📂 View Retrieved Qdrant Context Chunks (Ground Truth)"):
                for chunk in context_chunks:
                    st.write(f"• {chunk}")

        # Step 2: Formulate explicit system directives to prevent model hallucination
        system_instruction = (
            "You are an official AI Assistant for GL Bajaj Institute of Technology and Management (GLBITM). "
            "Your job is to answer the student's question accurately using only the verified documentation context provided. "
            "If the answer cannot be found within the provided context, politely state that you do not have that "
            "official data available from your scraped links, and offer general placeholder guidance. Keep your tone helpful and professional."
        )

        prompt_content = f"""
        VERIFIED DOCUMENTATION CONTEXT:
        {combined_context}

        STUDENT QUESTION:
        {user_input}
        """

        # Step 3: Dispatch instructions and structural payload to the modern Gemini backend engine
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt_content,
                config={"system_instruction": system_instruction}
            )
            st.markdown("### 🤖 Assistant Response:")
            st.write(response.text)
        except Exception as e:
            st.error(f"❌ Gemini Generation Failed: {e}")