from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

client = QdrantClient(
    url=os.getenv("QDRANT_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY")
)
print(client.count("local_movie_db", exact=True))
