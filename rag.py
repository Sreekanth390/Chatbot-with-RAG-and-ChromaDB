import chromadb
import ollama

# Create Persistent ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
collection = chroma_client.get_or_create_collection(
    name="company_documents"
)


def ask_question(question):

    # --------------------------------
    # 1. Create embedding for question
    # --------------------------------
    response = ollama.embed(
        model="nomic-embed-text",
        input=question
    )

    question_embedding = response["embeddings"][0]

    # --------------------------------
    # 2. Search ChromaDB
    # --------------------------------
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    print("Results:", results)

    # --------------------------------
    # 3. Get retrieved documents
    # --------------------------------
    documents = results["documents"][0]

    print("Documents:", documents)

    # --------------------------------
    # 4. Combine documents into context
    # --------------------------------
    context = "\n\n".join(documents)

    print("Context:", context)

    # --------------------------------
    # 5. Create RAG prompt
    # --------------------------------
    prompt = f"""
Answer the question using only the context provided below.

If the answer cannot be found in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}
"""

    # --------------------------------
    # 6. Generate answer using Llama
    # --------------------------------
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]



