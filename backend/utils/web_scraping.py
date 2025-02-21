from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

# Load HTML
from langchain_community.document_loaders import AsyncHtmlLoader

urls = ["https://genestory.ai/home/ve-chung-toi/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()
# Transform

print(docs[0].page_content)
bs_transformer = BeautifulSoupTransformer()

docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["span"])
# Result


print(docs_transformed[0].page_content[:])
# store the transformed data in a variable
with open("output.txt", "w") as f:
    f.write(docs_transformed[0].page_content)


    