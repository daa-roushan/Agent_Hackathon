from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# -----------------------------
# 1️⃣ Load environment variables
# -----------------------------
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("API_VERSION")
AZURE_CHAT_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME")       # e.g. gpt-4o-deploy
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")  # e.g. text-embedding-3-large

# -----------------------------
# 2️⃣ Initialize Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# 3️⃣ Load FAISS vectorstore
# -----------------------------
VECTORSTORE_PATH = os.path.join("model", "vectorstore/faiss_store")

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
)

print("🔹 Loading FAISS vectorstore...")
vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Vectorstore loaded successfully!")

# -----------------------------
# 4️⃣ Initialize Azure OpenAI LLM
# -----------------------------
llm = AzureChatOpenAI(
    azure_deployment=AZURE_CHAT_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0
)

# -----------------------------
# 5️⃣ Create RetrievalQA Chain
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# -----------------------------
# 6️⃣ Define Query Endpoint
# -----------------------------
@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        query_text = data.get("query", "")
        if not query_text:
            return jsonify({"error": "No query provided"}), 400

        print(f"🧠 Received query: {query_text}")
        response = qa_chain.run(query_text)
        print(f"✅ Response generated")

        return jsonify({"response": response})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------
# 7️⃣ Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
