# ingest.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader

loader = DirectoryLoader("./data", glob="**/*.txt")
docs = loader.load()
embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
vectorstore.persist()
print("Ingest Done")