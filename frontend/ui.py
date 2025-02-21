import os
import logging
import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader

def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = PDFPlumberLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error("PDF file not found.")
        return None


# Streamlit UI
st.title("📄 IELTS Chatbot - Upload Your Answer Sheet")

uploaded_file = st.file_uploader("Upload your IELTS Answer PDF", type=["pdf"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    pdf_text = ingest_pdf(uploaded_file)
    st.text_area("Extracted Text:", pdf_text, height=300)

# Chat Interface
st.subheader("💬 Chat with IELTS AI Agent")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    message(msg["text"], is_user=msg["is_user"])

user_input = st.text_input("You:", "")
if st.button("Send") and user_input:
    st.session_state["messages"].append({"text": user_input, "is_user": True})
    response = "This is a placeholder response from AI."  # TODO: Integrate AI response
    st.session_state["messages"].append({"text": response, "is_user": False})
    st.rerun()
