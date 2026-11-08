import os
from dotenv import load_dotenv

load_dotenv(override=True)

# LLM (지금은 Gemini)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash-lite"  # 안 되면 AI Studio에서 현재 쓸 수 있는 모델명 확인

# 임베딩 / DB
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION = "tactics"