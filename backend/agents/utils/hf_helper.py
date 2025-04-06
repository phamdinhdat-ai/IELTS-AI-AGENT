import os
import json 
import logging 
import pathlib
from pathlib import Path
import yaml
from typing import Any, Dict, List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint


def load_embeddings(embedding_path:str): # loadd embeddings from huggingface
    embdding = HuggingFaceEmbeddings(model_name = embedding_path)
    return embdding

def load_model(model_hf_path: str, hf_api_token:str):
    llm = HuggingFaceEndpoint(
        repo_id=model_hf_path,
        huggingfacehub_api_token=hf_api_token
        
    )
    return llm
