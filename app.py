import streamlit as st
import os
from google.genai import Client
from google.genai import types
from dotenv import load_dotenv
load_dotenv() 

 
from core.retriever import CollegeKnowledgeRetriever
from core.classifier import collegeChunkClassifier


st.set_page_config(
    page_title=" Gl bajaj College Intelligent Assistant",
    page_icon="🎓",
    layout="centered"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 2rem; }
    .context-box { background-color: #F3F4F6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1E3A8A; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓 College Smart RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Instant, verified answers regarding courses, admissions, and campus regulations.</div>', unsafe_allow_html=True)

@st.cache_resource
def initialize_core_components():
    """Initializes the Gemini Client and RAG modules once and caches them."""

    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("❌ Critical Error: GEMINI_API_KEY not found in Streamlit Secrets or Environment.")
        st.stop()
   
    genai_client = Client(api_key=api_key)
    

    retriever = CollegeKnowledgeRetriever()
    classifier = collegeChunkClassifier()
    
    return genai_client, retriever, classifier

try:
    client, retriever, classifier = initialize_core_components()
except Exception as e:
    st.error(f"Failed to load core database or models: {str(e)}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI campus guide. How can I assist you with college updates today?"}
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask about admissions, fees, courses, or timings..."):

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("Analyzing campus knowledge base..."):
            try:
                
                query_intent = classifier.predict_category(user_query)
                
              
                retrieved_docs = retriever.search_knowledge_base(user_query, intent=query_intent)
                context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
                system_instruction = (
                    "You are an official, supportive AI Chatbot representing the college administration.\n"
                    "Analyze the verified context provided below to formulate a concise, professional answer.\n"
                    "If the answer cannot be found within the provided context chunks, politely inform the user "
                    "that specific information isn't in your current system and advise contacting the campus helpdesk."
                )
                
                user_content = f"Context Material:\n{context_str}\n\nUser Question: {user_query}"
                
                
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3, 
                        max_output_tokens=800
                    )
                )
                
                assistant_response = response.text
                response_placeholder.markdown(assistant_response)
                
                
                if retrieved_docs:
                    with st.expander("📚 View Verified Knowledge Sources"):
                        for idx, doc in enumerate(retrieved_docs, start=1):
                            source_info = doc.metadata.get("source", "Campus Database")
                            st.markdown(f"**Source {idx} ({source_info}):**")
                            st.markdown(f"*{doc.page_content}*")
                            st.write("---")

           
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
            except Exception as error:
                error_msg = f"An execution boundary error occurred: {str(error)}"
                response_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})