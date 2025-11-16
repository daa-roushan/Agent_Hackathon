# ====================================================
# 🖼️ IMAGE ANALYSIS AND EMBEDDING PIPELINE (AZURE)
# Supports multiple store folders
# ====================================================

import os
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ----------------------------------------------------
# 1️⃣ Load environment variables
# ----------------------------------------------------
load_dotenv()

AZURE_EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
OPENAI_API_VERSION = os.getenv("API_VERSION")

IMAGE_BASE_DIR = "../data/images"
VECTORSTORE_PATH = "../backend/model/vectorstore"

# ----------------------------------------------------
# 2️⃣ Initialize embedding model
# ----------------------------------------------------
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    openai_api_version=OPENAI_API_VERSION
)

# ----------------------------------------------------
# 3️⃣ OCR / caption function
# ----------------------------------------------------
def extract_text_from_image(image_path):
    """
    Uses pytesseract OCR to extract visible text.
    You can replace this with a Vision model later.
    """
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        if not text.strip():
            text = f"Visual analysis placeholder for {os.path.basename(image_path)}"
        return text.strip()
    except Exception as e:
        print(f"⚠️ Error reading {image_path}: {e}")
        return ""

# ----------------------------------------------------
# 4️⃣ Process all stores
# ----------------------------------------------------
all_docs = []

store_folders = [f for f in os.listdir(IMAGE_BASE_DIR) if os.path.isdir(os.path.join(IMAGE_BASE_DIR, f))]

for store_name in store_folders:
    store_path = os.path.join(IMAGE_BASE_DIR, store_name)
    image_files = [os.path.join(store_path, f) for f in os.listdir(store_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"🏪 Processing images for {store_name}: {len(image_files)} files")

    for img_path in image_files:
        text_data = extract_text_from_image(img_path)
        doc = Document(page_content=text_data, metadata={"store": store_name, "source": img_path})
        all_docs.append(doc)

print(f"✅ Total images processed: {len(all_docs)}")

# ----------------------------------------------------
# 5️⃣ Split text into chunks
# ----------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(all_docs)

# ----------------------------------------------------
# 6️⃣ Load existing FAISS or create new
# ----------------------------------------------------
if os.path.exists(VECTORSTORE_PATH):
    print("📦 Loading existing vectorstore...")
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    vectorstore.add_documents(split_docs)
else:
    print("🆕 Creating new vectorstore")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

# ----------------------------------------------------
# 7️⃣ Save updated FAISS
# ----------------------------------------------------
vectorstore.save_local(VECTORSTORE_PATH)
print(f"💾 Saved vectorstore with image data for {len(store_folders)} stores at: {VECTORSTORE_PATH}")

# ----------------------------------------------------
# 8️⃣ Optional: test query
# ----------------------------------------------------
query = "Which store has the cleanest shelves?"
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.get_relevant_documents(query)

print("\n🔍 Sample retrieval results:")
for i, doc in enumerate(results, 1):
    print(f"{i}. [{doc.metadata['store']}] {doc.page_content[:150]}...\n")
