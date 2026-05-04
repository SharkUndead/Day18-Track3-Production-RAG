import qdrant_client
print(f"File: {qdrant_client.__file__}")
from qdrant_client import QdrantClient
client = QdrantClient(location=":memory:")
print(f"Client type: {type(client)}")
print(f"Has search: {hasattr(client, 'search')}")
