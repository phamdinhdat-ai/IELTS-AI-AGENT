import os
import logging 
from typing import  List,Iterator
from langchain_core.document_loaders import  BaseLoader 
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from docling.document_converter import DocumentConverter
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
        data = loader.lazy_load()
        return data
    else:
        logging.error(f"PDF file not found at path: {file_path}")
        return None
    
start_time = time.time()
file_pdf = "data/Pham Dinh Dat - RESUME - GST.pdf"
data = load_pdf(file_pdf)
for doc in data:
    print(doc.page_content)
    print("\n")
    print("===============================================")
    print("\n")
end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds.")