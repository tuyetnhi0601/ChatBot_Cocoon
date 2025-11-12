from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, json
from pathlib import Path
from infer_chatbot import respond  # dùng hàm có sẵn trong infer_chatbot.py

# ======== Load model và rules ========
artifacts_dir = Path("artifacts")

if not artifacts_dir.exists():
    raise FileNotFoundError("❌ Không tìm thấy thư mục artifacts. Hãy chạy train.py trước.")

with open("rules.json", "r", encoding="utf-8") as f:
    RULES = json.load(f)

# ======== Khởi tạo FastAPI ========
app = FastAPI(title="CocoBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "CocoBot API đang hoạt động 🌿"}

@app.post("/chat")
def chat(req: ChatRequest):
    result = respond(req.message, artifacts_dir, RULES)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
