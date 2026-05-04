from qdrant_client import QdrantClient
client = QdrantClient(location=":memory:")
print(f"Attributes: {dir(client)}")
if hasattr(client, 'search'):
    print("Found 'search' method")
else:
    print("NOT Found 'search' method")
