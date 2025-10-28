from chromadb import PersistentClient

c = PersistentClient(path="vectorstore")
col = c.get_collection("python_course_docs")
print("✅ Collection:", col.name)
print("✅ Total vectors:", col.count())
