from chromadb import PersistentClient
import os

VSTORE_DIR = "vectorstore"

client = PersistentClient(path=VSTORE_DIR)
collections = client.list_collections()

print("✅ Collections found:", [c.name for c in collections])

if collections:
    col = client.get_collection(collections[0].name)
    print("Total vectors in collection:", col.count())
    sample = col.peek(2)
    for i, doc in enumerate(sample["documents"]):
        print(f"\n--- Sample {i+1} ---")
        print("Source:", sample["metadatas"][i]["source"])
        print("Preview:", doc[:200], "...")
else:
    print("No collections found.")

print("Resolved vectorstore path:", os.path.abspath(VSTORE_DIR))
