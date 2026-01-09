import os
import json
import shutil
import tempfile
import uuid
import urllib.request
from typing import List, Union

# 尝试导入 Qdrant，如果环境有问题也能捕捉到
try:
    import qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_VERSION = qdrant_client.__version__
except ImportError:
    QDRANT_VERSION = "Not Installed"
    QdrantClient = None

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. 初始化配置 ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "123456")

client = OpenAI(api_key=OPENAI_API_KEY)

# 初始化 Qdrant 客户端
if QdrantClient:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    qdrant = None
    print("⚠️ Warning: Qdrant client not installed properly.")

COLLECTION_NAME = "teachers_skills"

app = FastAPI(title="PandaFreeAI Engine (Stable)")

# 允许跨域
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. 启动检查 ---
@app.on_event("startup")
def startup_event():
    if qdrant:
        try:
            if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
                qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                print(f"Collection {COLLECTION_NAME} created.")
        except Exception as e:
            print(f"Startup Warning (Non-fatal): {str(e)}")

# --- 3. 核心工具函数 ---

def get_embedding(text: str):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def analyze_audio_transcript(transcript: str):
    # 你的 GPT 逻辑保持不变
    system_prompt = """
    You are an expert IELTS Speaking examiner. Analyze the transcript.
    Return JSON with: 'overall_score', 'feedback', 'weakness_search_query'.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript}
        ]
    )
    return json.loads(response.choices[0].message.content)

def safe_qdrant_search(query_vector, limit=3):
    """
    🛡️ Qdrant 万能搜索函数
    自动适配不同版本的 qdrant-client，彻底解决 'no attribute search' 问题
    """
    if not qdrant:
        return []

    # 方案 A: 标准新版 (v1.0+)
    if hasattr(qdrant, 'search'):
        print("Using standard 'search' method")
        return qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )
    
    # 方案 B: 旧版兼容 (v0.11 等)
    elif hasattr(qdrant, 'search_points'):
        print("Using legacy 'search_points' method")
        return qdrant.search_points(
            collection_name=COLLECTION_NAME,
            vector=query_vector,
            limit=limit
        )
        
    # 方案 C: 更加古老的版本
    else:
        print(f"🚨 Critical: No search method found in Qdrant version {QDRANT_VERSION}")
        return []

# --- 4. API 接口 ---

# 新增：版本检查接口 (用于调试)
@app.get("/")
def home():
    return {"status": "running", "qdrant_version": QDRANT_VERSION}

@app.post("/assess-audio")
async def assess_audio(file: Union[UploadFile, str] = File(...)):
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        # 下载/保存音频
        if isinstance(file, str):
            req = urllib.request.Request(file, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(temp_file_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        else:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # 1. Whisper 转录
        with open(temp_file_path, "rb") as audio:
            transcription = client.audio.transcriptions.create(model="whisper-1", file=audio)
        transcript_text = transcription.text

        # 2. GPT 评分
        ai_result = analyze_audio_transcript(transcript_text)
        
        # 3. 向量搜索 (使用万能函数)
        query_vector = get_embedding(ai_result['weakness_search_query'])
        search_result = safe_qdrant_search(query_vector)
        
        # 4. 格式化结果
        recommended_teachers = []
        for hit in search_result:
            # 兼容不同版本的 payload 访问方式
            payload = hit.payload if hasattr(hit, 'payload') else hit.get('payload', {})
            recommended_teachers.append({
                "bubble_id": payload.get('bubble_id', 'unknown'),
                "name": payload.get('name', 'Unknown Teacher'),
                "match_score": hit.score,
                "specialty": payload.get('specialty', '')
            })

        return {
            "status": "success",
            "transcript": transcript_text,
            "score": ai_result['overall_score'],
            "feedback": ai_result['feedback'],
            "recommendations": recommended_teachers
        }

    except Exception as e:
        # 打印错误并返回给 Bubble
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)} | Qdrant Ver: {QDRANT_VERSION}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Admin 接口 (带 Form)
@app.post("/admin/add-teacher")
async def add_teacher(name: str = Form(...), specialty_desc: str = Form(...), bubble_id: str = Form(...), secret_key: str = Header(None)):
    if secret_key != ADMIN_SECRET: raise HTTPException(status_code=401)
    vector = get_embedding(specialty_desc)
    
    point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"bubble_id": bubble_id, "name": name, "specialty": specialty_desc})
    
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    # 🟢 修复点：自动读取 Render 分配的端口，如果没有则默认 10000
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
