# ====================================================
# 🎥 VIDEO ANALYSIS AND EMBEDDING PIPELINE (AZURE)
# Supports multiple store folders
# ====================================================

import os
import cv2
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------------------------------
# 1️⃣ Load environment variables
# ----------------------------------------------------
load_dotenv()

AZURE_EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
OPENAI_API_VERSION = os.getenv("API_VERSION")

VIDEO_BASE_DIR = "../data/videos"
VECTORSTORE_PATH = "../backend/model/vectorstore"

# ----------------------------------------------------
# 2️⃣ Initialize embeddings
# ----------------------------------------------------
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    openai_api_version=OPENAI_API_VERSION
)

# ----------------------------------------------------
# 3️⃣ Frame extraction + description
# ----------------------------------------------------
def extract_frames(video_path, frame_interval=30):
    """
    Extracts frames every 30 seconds.
    """
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(frame_interval * fps) if fps > 0 else 60
    count, saved_frames = 0, []

    while True:
        success, image = vidcap.read()
        if not success:
            break
        if count % interval == 0:
            frame_path = f"{video_path}_frame_{count}.jpg"
            cv2.imwrite(frame_path, image)
            saved_frames.append(frame_path)
        count += 1

    vidcap.release()
    return saved_frames

def describe_frame(frame_path):
    """
    Uses OCR to get text — can be replaced with Vision model for richer context.
    """
    try:
        img = Image.open(frame_path)
        text = pytesseract.image_to_string(img)
        if not text.strip():
            text = f"Frame {os.path.basename(frame_path)} shows store layout or customers."
        return text.strip()
    except Exception as e:
        print(f"⚠️ Error reading frame {frame_path}: {e}")
        return ""

# ----------------------------------------------------
# 4️⃣ Process all stores
# ----------------------------------------------------
all_docs = []
store_folders = [f for f in os.listdir(VIDEO_BASE_DIR) if os.path.isdir(os.path.join(VIDEO_BASE_DIR, f))]

for store_name in store_folders:
    store_path = os.path.join(VIDEO_BASE_DIR, store_name)
    video_files = [os.path.join(store_path, f) for f in os.listdir(store_path) if f.lower().endswith(('.mp4', '.mov', '.avi'))]

    print(f"🏪 Processing videos for {store_name}: {len(video_files)} files")

    for video_path in video_files:
        frames = extract_frames(video_path, frame_interval=30)
        for frame_path in frames:
            desc = describe_frame(frame_path)
            doc = Document(page_content=desc, metadata={"store": store_name, "video": os.path.basename(video_path)})
            all_docs.append(doc)

print(f"✅ Total frame documents created: {len(all_docs)}")

# ----------------------------------------------------
# 5️⃣ Split and embed
# ----------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(all_docs)

if os.path.exists(VECTORSTORE_PATH):
    print("📦 Loading existing vectorstore...")
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    vectorstore.add_documents(split_docs)
else:
    print("🆕 Creating new vectorstore")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

# ----------------------------------------------------
# 6️⃣ Save
# ----------------------------------------------------
vectorstore.save_local(VECTORSTORE_PATH)
print(f"💾 Saved vectorstore with video data for {len(store_folders)} stores at: {VECTORSTORE_PATH}")

# ----------------------------------------------------
# 7️⃣ Optional query
# ----------------------------------------------------
query = "Which store seems the most crowded?"
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.get_relevant_documents(query)

print("\n🔍 Sample retrieval results:")
for i, doc in enumerate(results, 1):
    print(f"{i}. [{doc.metadata['store']}] {doc.page_content[:150]}...\n")
