# ====================================================
# 📊 DATA PREPROCESSING AND VECTORSTORE CREATION (AZURE)
# ====================================================

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader, UnstructuredExcelLoader

# ----------------------------------------------------
# 1️⃣ Load environment variables
# ----------------------------------------------------
load_dotenv()

# AZURE_EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
# OPENAI_API_VERSION = os.getenv("API_VERSION")

DATA_DIR = "../data/excel"
VECTORSTORE_DIR = "../backend/model/vectorstore"

os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# ----------------------------------------------------
# 2️⃣ Collect all data files (Excel + CSV)
# ----------------------------------------------------
data_files = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith((".xlsx", ".xls", ".csv"))
]

all_docs = []

for file_path in data_files:
    print(f"📂 Loading file: {file_path}")
    try:
        if file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        else:
            loader = UnstructuredExcelLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)
    except Exception as e:
        print(f"⚠️ Skipping {file_path}: {e}")

print(f"✅ Loaded {len(all_docs)} documents from {len(data_files)} files")

# ----------------------------------------------------
# 3️⃣ Split documents into smaller chunks
# ----------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(all_docs)
print(f"📑 Split into {len(split_docs)} text chunks")

# ----------------------------------------------------
# 4️⃣ Create embeddings using Azure OpenAI
# ----------------------------------------------------
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
OPENAI_API_VERSION = os.getenv("API_VERSION")

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=OPENAI_API_VERSION
)

vectorstore = FAISS.from_documents(split_docs, embeddings)

# ----------------------------------------------------
# 5️⃣ Save FAISS vectorstore
# ----------------------------------------------------
vectorstore.save_local(VECTORSTORE_DIR)
print(f"💾 Saved vectorstore to: {VECTORSTORE_DIR}")

# ----------------------------------------------------
# 6️⃣ Optional — Test retrieval
# ----------------------------------------------------
test_query = "Which store had the highest revenue or sales performance?"
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.get_relevant_documents(test_query)

print("\n🔍 Sample retrieval results:")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content[:200]}...\n")
