from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

DATA_PATH = r"C:\Users\User\RAG_Task1\data set"
CHROMA_PATH = r"chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection =chroma_client.get_or_create_collection(name="growing_vegetables")


loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
chunk_size=300,
chunk_overlap=100,
length_function=len,
is_separator_regex=False,
)
chunks = text_splitter.split_documents(raw_documents)




documents = [chunk.page_content for chunk in chunks]
metadata = [chunk.metadata for chunk in chunks]
ids = ["ID"+str(i) for i in range(len(chunks))]


collection.upsert(
documents=documents,
metadatas=metadata,
ids=ids
)

