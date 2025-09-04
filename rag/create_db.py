from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-06-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

retriever = WikipediaRetriever(
    top_k_results=1,
    lang="en",
)

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY")
)

for lang_code in ["en", "ro"]:
    collection_name = f"local_movie_db_{lang_code}"
    if not qdrant_client.collection_exists(collection_name=collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
qdrant_db = QdrantVectorStore(
    client=qdrant_client, collection_name="local_movie_db", embedding=embeddings
)
movie_docs = []
movies = ["Inception", "The Return of the King", "Shutter Island", "The Dark Knight"]

# for movie in movies:
#     movie_docs += retriever.invoke(movie)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# movie_docs_split = text_splitter.split_documents(movie_docs)

# movie_vector_db = qdrant_db.add_documents(documents=movie_docs_split)

def ingest_movies(lang_code, movies):
    retriever = WikipediaRetriever(
        top_k_results=1,
        lang=lang_code,  # "en" or "ro"
    )
    qdrant_db = QdrantVectorStore(
        client=qdrant_client,
        collection_name=f"local_movie_db_{lang_code}",
        embedding=embeddings,
    )

    all_docs = []
    for movie in movies:
        all_docs += retriever.invoke(movie)

    split_docs = text_splitter.split_documents(all_docs)
    qdrant_db.add_documents(documents=split_docs)

ingest_movies("en", movies)
ingest_movies("ro", movies)