import os
import logging 
from typing import  List,Iterator, Any
from langchain_core.document_loaders import  BaseLoader 
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from langchain_community.vectorstores import FAISS
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFMinerLoader
import ollama 
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
# from langchain_text_splitters.word import RecursiveWordTextSplitter
import time 
class DocumentPDFLoader(BaseLoader):
    
    def __init__(self, filepath: List[str]) -> None: 
        self._filepath = filepath if isinstance(filepath, list) else [filepath]
        self._coverter = DocumentConverter()
    
    def lazy_load (self)->Iterator[Document]:
        for file in self._filepath:
            dl = self._coverter.convert(file).document
            text = dl.export_to_markdown()
            yield Document(page_content=text)


def load_pdf(file_path:str)->str:
    if os.path.exists(file_path):
        loader = DocumentPDFLoader(file_path)
        data = loader.load()
        return data
    else:
        logging.error(f"PDF file not found at path: {file_path}")
        return None
    
# start_time = time.time()
file_pdf = "data/ielts_listening_practice_test_pdf_1_1_1ae068b05d.pdf"
# data = load_pdf(file_pdf)
# for doc in data:
#     print(doc.page_content)
#     print("\n")
#     print("===============================================")
#     print("\n")
# end_time = time.time()

# print(f"Time taken: {end_time - start_time} seconds.")

def load_pdf_langchain(file_path:str)->str:
    if os.path.exists(file_path):
        loader = PDFMinerLoader(file_path, headers=None)
        data = loader.load()
        return data
    else:
        logging.error(f"PDF file not found at path: {file_path}")
        return None


def document_chunking(docs : List[str], chunker) -> List[str]:
    text_chunks  = chunker.split_documents(docs)
    return text_chunks


def create_vectorstore(text_chunks:List[str], embeddings) -> Any:
    return FAISS.from_documents(text_chunks, embedding=embeddings)

data = load_pdf_langchain(file_pdf)
start_time = time.time()
str_data = ""

for doc in data:
    text = doc.page_content
    str_data += text
    print(doc.page_content)
    print("\n")
    print("===============================================")
    print("\n")
end_time = time.time()
with open("data/ielts_listening_practice_test_pdf_1_1_1ae068b05d.txt", "w") as f:
    f.write(str_data)
print(f"Time taken: {end_time - start_time} seconds.")


start_time = time.time()
chunker = RecursiveCharacterTextSplitter(chunk_size=512 ,  chunk_overlap=100)
document_chunks = document_chunking(data,chunker)
end_time = time.time()
print("Time taken to split the document: ", end_time - start_time)
for chunk in document_chunks:
    print(chunk)
    print("\n")
    print("===============================================")
    print("\n")

# def split_documents(documents, chunk_size=1000, chunk_overlap=20):
#     text_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     chunks = text_split.split_documents(documents)
#     return chunks

print(f"Total number of chunks: {len(document_chunks)}")
print("\n")
embeddings = OllamaEmbeddings(model="nomic-embed-text", temperature=0.4)

db = Chroma.from_documents(document_chunks, embeddings, collection_name = "local-rag")

retriever = db.as_retriever()

template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question:
If you don't know the answer, then answer from your own knowledge and dont give just one word answer, and dont tell the user that you are answering from your knowledge.
Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:

"""
prompt = ChatPromptTemplate.from_template(template)

local_model = "deepseek-r1:1.5b" #Enter your preferred model here
llm = ChatOllama(model=local_model)

rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

while True:
    query = str(input("Enter Question: "))
    print(rag_chain.invoke(query))