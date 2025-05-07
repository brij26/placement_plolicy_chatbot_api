from fastapi import FastAPI
from pydantic import BaseModel
import cassio
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Cassandra
import httpx
from fastapi.middleware.cors import CORSMiddleware
import os



# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ['ASTRA_DB_ID']=os.getenv("ASTRA_DB_ID")
os.environ['ASTRA_DB_APPLICATION_TOKEN']=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
# Step 1: Initialize Astra DB with CassIO
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_KEYSPACE = "default_keyspace"

cassio.init(
    database_id=ASTRA_DB_ID,
    token=ASTRA_DB_APPLICATION_TOKEN,
    keyspace=ASTRA_KEYSPACE
)

# Step 2: Set up embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Cassandra(
    embedding=embedding_model,
    table_name="minipoject_1",
    session=None,
    keyspace="default_keyspace",
)

# Step 3: Groq API setup
GROQ_API_KEY = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
GROQ_MODEL = "Llama3-8b-8192"

# Step 4: Define request schema
class QuestionInput(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(input: QuestionInput):
    query = input.question

    # Step 5: Perform similarity search from Astra DB
    try:
        retrieved_docs = vector_store.similarity_search(query, k=5)
    except Exception as e:
        return {"error": f"Vector search failed: {str(e)}"}

    if not retrieved_docs:
        return {"answer": "Sorry, I couldn't find any relevant information."}

    # Step 6: Create detailed prompt for full answer extraction
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"""

    # Step 7: Query Groq API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a precise and detailed assistant. Extract information exactly from the provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0  # Lower temperature = more factual, less creative
                }
            )
        data = response.json()
        # print("DEBUG - Full Groq API response:", data)
        answer = data["choices"][0]["message"]["content"]
    except Exception as e:
        return {"error": f"Groq API failed: {str(e)}"}

    return {"answer": answer}
