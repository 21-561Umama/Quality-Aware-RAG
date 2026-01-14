import streamlit as st
from rag_engine import RAGEngine
import os

st.set_page_config(page_title="Quality-Aware RAG", layout="wide")
st.title("🛡️ Quality-Aware RAG System")

# Initialize Engine in session state
if 'engine' not in st.session_state:
    st.session_state.engine = RAGEngine()

# Sidebar for Uploads
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Indexing document..."):
            chunks = st.session_state.engine.process_pdf("temp.pdf")
            st.success(f"Indexed {chunks} chunks!")

# Chat Interface
query = st.chat_input("Ask a question about your document...")
if query:
    with st.chat_message("user"):
        st.write(query)
        
    with st.chat_message("assistant"):
        answer, faith, conf = st.session_state.engine.get_response(query)
        st.write(answer)
        
        # Displaying the Quality Metrics
        col1, col2 = st.columns(2)
        col1.metric("Faithfulness Score", f"{faith*100:.0f}%")
        col2.metric("Confidence Score", f"{conf*100:.0f}%")
        
        if faith < 0.7:
            st.warning("⚠️ Warning: This answer may contain hallucinations.")
