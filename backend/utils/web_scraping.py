from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

# Load HTML
from langchain_community.document_loaders import AsyncHtmlLoader

urls = ["https://genestory.ai/home/ve-chung-toi/"]
loader = AsyncChromiumLoader(urls )
docs = loader.load()
# Transform

print(docs[0].page_content)
bs_transformer = BeautifulSoupTransformer()

docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["span"], remove_comments=True)
# Result
for doc in docs_transformed:
    print(doc.page_content)



    