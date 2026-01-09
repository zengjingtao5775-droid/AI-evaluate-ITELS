import os
import json
import shutil
import tempfile
import uuid
import urllib.request
from typing import List, Union

# 1. 导入核心库
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

# 2. 尝试导入 Qdrant 并获取版本号 (侦探代码)
try:
    import qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    INSTALLED_VERSION = qdrant_client.__version__
except ImportError:
    QdrantClient = None
    INSTALLED_VERSION = "NOT_INSTALLED"

# 3. 初始化
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "123456")

client = OpenAI(api_key=OPENAI_API_KEY)

# 安全初始化 Qdrant
if QdrantClient:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    qdrant = None

COLLECTION_NAME = "teachers_skills"

app = FastAPI(title="PandaFreeAI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 新增：调试接口 (Debug Endpoint) ---
@app.get("/debug-version")
def debug_version():
    """
    这个接口专门用来查案，看看 Render 到底装了哪个版本
    """
    return {
        "status": "online",
        "qdrant_version": INSTALLED_VERSION,
        "has_search_method": hasattr(qdrant, 'search') if qdrant else False,
        "has_search_points_method": hasattr(qdrant, 'search_points') if qdrant else False
    }

@app.on_event("startup")
def startup_event():
    print(f"🚀 Server Starting... Installed Qdrant Version: {INSTALLED_VERSION}")
    if qdrant:
        try:
            # 兼容旧版本检查 collection 的方法
            if hasattr(qdrant, 'collection_exists'):
                exists = qdrant.collection_exists(collection_name=COLLECTION_NAME)
            else:
                # 极旧版本可能需要 get_collection
                try:
                    qdrant.get_collection(COLLECTION_NAME)
                    exists = True
                except:
                    exists = False
            
            if not exists:
                qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                print(f"Collection {COLLECTION_NAME} created.")
        except Exception as e:
            print(f"Startup check warning: {e}")

# --- 工具函数 ---
def get_embedding(text: str):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def analyze_audio_transcript(transcript: str):
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

# --- 🛡️ 核心修复：万能搜索函数 ---
def safe_qdrant_search(query_vector, limit=3):
    if not qdrant:
        print("❌ Qdrant client is missing.")
        return []

    # 1. 尝试新版 search (v1.0+)
    if hasattr(qdrant, 'search'):
        print(f"✅ Using standard 'search' (Version: {INSTALLED_VERSION})")
        return qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )
    
    # 2. 尝试旧版 search_points (v0.x)
    elif hasattr(qdrant, 'search_points'):
        print(f"⚠️ Using legacy 'search_points' (Version: {INSTALLED_VERSION})")
        return qdrant.search_points(
            collection_name=COLLECTION_NAME,
            vector=query_vector,
            limit=limit
        )
    
    # 3. 如果都不行，返回空列表，防止报错崩掉
    else:
        print(f"🚨 Critical: No search method found. Please update library.")
        return []

@app.post("/assess-audio")
async def assess_audio(file: Union[UploadFile, str] = File(...)):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        if isinstance(file, str):
            req = urllib.request.Request(file, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(temp_file_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        else:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # 1. Whisper
        with open(temp_file_path, "rb") as audio:
            transcription = client.audio.transcriptions.create(model="whisper-1", file=audio)
        transcript_text = transcription.text

        # 2. GPT
        ai_result = analyze_audio_transcript(transcript_text)
        
        # 3. RAG 搜索 (⚠️ 关键：这里调用的是 safe_qdrant_search)
        query_vector = get_embedding(ai_result['weakness_search_query'])
        search_result = safe_qdrant_search(query_vector)
        
        recommended_teachers = []
        for hit in search_result:
            # 兼容旧版 payload 获取方式
            payload = hit.payload if hasattr(hit, 'payload') else hit.get('payload', {})
            recommended_teachers.append({
                "bubble_id": payload.get('bubble_id'),
                "name": payload.get('name'),
                "match_score": hit.score,
                "specialty": payload.get('specialty')
            })

        return {
            "status": "success",
            "transcript": transcript_text,
            "score": ai_result['overall_score'],
            "feedback": ai_result['feedback'],
            "recommendations": recommended_teachers
        }

    except Exception as e:
        print(f"Error: {e}")
        # 把版本号返回给 Bubble，方便我们看
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)} | Installed Qdrant Ver: {INSTALLED_VERSION}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Admin 接口
@app.post("/admin/add-teacher")
async def add_teacher(name: str = Form(...), specialty_desc: str = Form(...), bubble_id: str = Form(...), secret_key: str = Header(None)):
    if secret_key != ADMIN_SECRET: raise HTTPException(status_code=401)
    vector = get_embedding(specialty_desc)
    point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"bubble_id": bubble_id, "name": name, "specialty": specialty_desc})
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
