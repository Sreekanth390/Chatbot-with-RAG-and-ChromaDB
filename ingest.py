import chromadb
import ollama

# Create Persistent ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
collection = chroma_client.get_or_create_collection(
    name="company_documents"
)

# Read document
with open("documents/company.txt", "r", encoding="utf-8") as file:
    document = file.read()

# Split document into chunks
chunks = document.split("\n\n")

# Generate embeddings using Ollama
for index, chunk in enumerate(chunks):

    response = ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )

    embedding = response["embeddings"][0]

    collection.add(
        ids=[f"chunk-{index}"],
        documents=[chunk],
        embeddings=[embedding]
    )

print("Document successfully stored in ChromaDB.")
