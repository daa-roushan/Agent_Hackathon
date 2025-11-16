# ====================================================
# 📊 INCREMENTAL DATA PREPROCESSING + VECTORSTORE (AZURE)
# ====================================================

import os
import time
import pickle
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader, UnstructuredExcelLoader
from openai import RateLimitError

# ----------------------------------------------------
# 1️⃣ Load environment variables
# ----------------------------------------------------
load_dotenv()

DATA_DIR = "../data/excel"
VECTORSTORE_DIR = "../backend/model/vectorstore"
VECTORSTORE_PATH = os.path.join(VECTORSTORE_DIR, "faiss_store")
PROCESSED_FILES_PATH = os.path.join(VECTORSTORE_DIR, "processed_files.pkl")

os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# ----------------------------------------------------
# 2️⃣ Initialize Azure embeddings
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

# ----------------------------------------------------
# 3️⃣ Load or create vectorstore
# ----------------------------------------------------
def load_vectorstore():
    if os.path.exists(VECTORSTORE_PATH):
        print("🔄 Loading existing vectorstore...")
        return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    print("🆕 Creating new vectorstore...")
    return None

# ----------------------------------------------------
# 4️⃣ Track processed files
# ----------------------------------------------------
def load_processed_files():
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, "rb") as f:
            return pickle.load(f)
    return set()

def save_processed_files(processed_files):
    with open(PROCESSED_FILES_PATH, "wb") as f:
        pickle.dump(processed_files, f)

# ----------------------------------------------------
# 🔹 NEW: Helper function — embed in batches with retries
# ----------------------------------------------------
def embed_in_batches(split_docs, batch_size=50, max_retries=2):
    """
    Embeds documents in smaller batches to handle Azure rate limits safely.
    Returns a merged FAISS vectorstore.
    """
    all_stores = []

    for i in range(0, len(split_docs), batch_size):
        batch = split_docs[i:i + batch_size]
        print(f"🧩 Embedding batch {i // batch_size + 1}/{(len(split_docs) + batch_size - 1) // batch_size} "
              f"({len(batch)} chunks)...")

        for attempt in range(max_retries):
            try:
                store = FAISS.from_documents(batch, embeddings)
                all_stores.append(store)
                break
            except RateLimitError:
                wait_time = 60 * (attempt + 1)
                print(f"⚠️ Rate limit hit on batch {i // batch_size + 1}. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        else:
            raise Exception(f"❌ Failed to embed batch {i // batch_size + 1} after {max_retries} retries.")

    # Merge all small FAISS stores into one
    main_store = all_stores[0]
    for store in all_stores[1:]:
        main_store.merge_from(store)

    return main_store

# ----------------------------------------------------
# 5️⃣ Main processing function (now uses batching)
# ----------------------------------------------------
def process_files():
    vectorstore = load_vectorstore()
    processed_files = load_processed_files()

    data_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith((".xlsx", ".xls", ".csv"))
    ]

    for file_path in data_files:
        file_name = os.path.basename(file_path)
        if file_name in processed_files:
            print(f"⏭️ Skipping already processed file: {file_name}")
            continue

        print(f"\n📂 Loading: {file_name}")
        try:
            if file_path.endswith(".csv"):
                loader = CSVLoader(file_path)
            else:
                loader = UnstructuredExcelLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = splitter.split_documents(docs)
            print(f"📑 Split into {len(split_docs)} chunks")

            # 🔹 CHANGED: replaced single FAISS.from_documents() with batched version
            new_store = embed_in_batches(split_docs, batch_size=50, max_retries=5)

            # Merge into main store
            if vectorstore is None:
                vectorstore = new_store
            else:
                vectorstore.merge_from(new_store)

            # Save updated store + processed file list
            vectorstore.save_local(VECTORSTORE_PATH)
            processed_files.add(file_name)
            save_processed_files(processed_files)

            print(f"✅ Successfully added embeddings from: {file_name}")

        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")

    print("\n🎯 All new files processed and vectorstore updated successfully.")

# ----------------------------------------------------
# 6️⃣ Run process
# ----------------------------------------------------
if __name__ == "__main__":
    process_files()
