
import os
import faiss

from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from google import genai


app = Flask(__name__)


# ==============================
# GEMINI
# ==============================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ==============================
# LOAD KNOWLEDGE
# ==============================

with open("knowledge.txt", "r", encoding="utf-8") as f:
    knowledge = f.read()


# ==============================
# CHUNKING
# ==============================

def create_chunks(text, chunk_size=800, overlap=150):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

    return chunks


chunks = create_chunks(knowledge)


# ==============================
# EMBEDDING MODEL
# ==============================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==============================
# CREATE EMBEDDINGS
# ==============================

embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")


# ==============================
# FAISS
# ==============================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# ==============================
# RETRIEVAL
# ==============================

def retrieve_documents(question, top_k=4):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for i in indices[0]:

        if i >= 0 and i < len(chunks):
            results.append(chunks[i])

    return results


# ==============================
# GENERATE ANSWER
# ==============================

def generate_answer(question):

    retrieved_chunks = retrieve_documents(
        question,
        top_k=4
    )

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are a Knowledge Transfer assistant.

Answer the question using ONLY the context
provided below.

If the answer is not available in the context,
say that the information was not found in the
knowledge document.

Do not invent information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==============================
# ASK API
# ==============================

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    question = data.get(
        "question",
        ""
    ).strip()

    if not question:

        return jsonify({
            "answer": "Please enter a question."
        })

    try:

        answer = generate_answer(
            question
        )

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        return jsonify({
            "answer": "Gemini API error: " + str(e)
        }), 500


# ==============================
# START SERVER
# ==============================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
