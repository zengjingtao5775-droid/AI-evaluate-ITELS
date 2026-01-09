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
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 2. 初始化
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "123456")

client = OpenAI(api_key=OPENAI_API_KEY)
# 直接初始化，不搞复杂的 try-except，报错就报错，方便定位
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION_NAME = "teachers_skills"

app = FastAPI(title="PandaFreeAI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    try:
        if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            print(f"Collection {COLLECTION_NAME} created.")
    except Exception as e:
        print(f"Startup check passed or skipped: {e}")

# --- 核心逻辑 ---

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

@app.post("/assess-audio")
async def assess_audio(file: Union[UploadFile, str] = File(...)):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        if isinstance(file, str):
            # 伪装 User-Agent 下载
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
        
        # 3. RAG 搜索
        query_vector = get_embedding(ai_result['weakness_search_query'])
        
        # 使用最标准的搜索方法
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
        
        recommended_teachers = []
        for hit in search_result:
            recommended_teachers.append({
                "bubble_id": hit.payload.get('bubble_id'),
                "name": hit.payload.get('name'),
                "match_score": hit.score,
                "specialty": hit.payload.get('specialty')
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
        raise HTTPException(status_code=500, detail=str(e))
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
    # 自动获取端口
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
