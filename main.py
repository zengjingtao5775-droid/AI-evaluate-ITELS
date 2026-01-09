import os
import json
import shutil
import tempfile
import uuid
import urllib.request
from typing import List, Union

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- 1. 初始化 ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "123456")

client = OpenAI(api_key=OPENAI_API_KEY)

# 尝试连接 Qdrant (包裹在 try-except 中以防崩坏)
qdrant = None
try:
    print(f"Connecting to Qdrant...")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    # 简单的连接测试
    print("Qdrant Client initialized.")
except Exception as e:
    print(f"⚠️ Warning: Qdrant connection failed: {e}")

COLLECTION_NAME = "teachers_skills"

app = FastAPI(title="PandaFreeAI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. 工具函数 ---

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

# --- 3. 核心接口 ---

@app.post("/assess-audio")
async def assess_audio(file: Union[UploadFile, str] = File(...)):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        # 1. 下载或保存文件
        if isinstance(file, str):
            print(f"Downloading URL: {file}")
            req = urllib.request.Request(file, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(temp_file_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        else:
            print("Saving uploaded file")
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # 2. Whisper 转录
        with open(temp_file_path, "rb") as audio:
            transcription = client.audio.transcriptions.create(model="whisper-1", file=audio)
        transcript_text = transcription.text

        # 3. GPT-4o 评分
        ai_result = analyze_audio_transcript(transcript_text)
        
        # 4. Qdrant 向量搜索
        recommended_teachers = []
        if qdrant:
            try:
                print("Searching Qdrant for teachers...")
                query_vector = get_embedding(ai_result['weakness_search_query'])
                
                # 直接搜索
                search_result = qdrant.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=3
                )
                
                for hit in search_result:
                    payload = hit.payload or {}
                    recommended_teachers.append({
                        "bubble_id": payload.get('bubble_id'),
                        "name": payload.get('name'),
                        "match_score": hit.score,
                        "specialty": payload.get('specialty')
                    })
            except Exception as e:
                print(f"Search error: {e}")
                # 搜索出错也不要崩，返回空列表
                recommended_teachers = []
        else:
            print("Qdrant not connected, skipping search.")

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

# --- 4. 添加老师接口 ---
@app.post("/admin/add-teacher")
async def add_teacher(name: str = Form(...), specialty_desc: str = Form(...), bubble_id: str = Form(...), secret_key: str = Header(None)):
    if secret_key != ADMIN_SECRET: raise HTTPException(status_code=401)
    
    if not qdrant:
        raise HTTPException(status_code=500, detail="Qdrant not connected")

    vector = get_embedding(specialty_desc)
    
    point = PointStruct(
        id=str(uuid.uuid4()), 
        vector=vector, 
        payload={
            "bubble_id": bubble_id, 
            "name": name, 
            "specialty": specialty_desc
        }
    )
    
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
    return {"status": "success", "message": f"Teacher {name} added."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
