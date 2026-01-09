import os
import json
import shutil
import tempfile
import uuid
import urllib.request # 用于下载 URL
from typing import List, Union # 确保引入 Union
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header # 🟢 引入 Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. 初始化配置
load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL") 
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "123456") # 默认密码

client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION_NAME = "teachers_skills"

app = FastAPI(title="PandaFreeAI Engine")

origins = ["*"] # 允许所有跨域，生产环境可改为你的域名

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"Collection {COLLECTION_NAME} created.")

# --- 工具函数 ---

def get_embedding(text: str):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def analyze_audio_transcript(transcript: str):
    system_prompt = """
    You are an expert IELTS Speaking examiner. Analyze the transcript.
    Return JSON with:
    - 'overall_score': Float (0-9)
    - 'feedback': String (Constructive feedback)
    - 'weakness_search_query': String (Describe the core problem for RAG search)
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

# --- API 接口 ---

# 1. 修复后的添加老师接口 (使用 Form)
@app.post("/admin/add-teacher")
async def add_teacher(
    name: str = Form(...), 
    specialty_desc: str = Form(...), 
    bubble_id: str = Form(...), 
    secret_key: str = Header(None)
):
    if secret_key != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    vector = get_embedding(specialty_desc)
    
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "bubble_id": bubble_id,
                    "name": name,
                    "specialty": specialty_desc
                }
            )
        ]
    )
    return {"status": "success", "teacher": name}

# 2. 修复后的评测接口 (兼容 URL 和 文件)
@app.post("/assess-audio")
async def assess_audio(file: Union[UploadFile, str] = File(...)):
    
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        # 判断是 URL 还是 文件流
        if isinstance(file, str):
            print(f"Received URL: {file}")
            # 伪装 Header 避免某些 CDN 拒绝 python-urllib 请求 (可选，但推荐)
            req = urllib.request.Request(
                file, 
                data=None, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response, open(temp_file_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        else:
            print("Received Binary File")
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # 正常业务逻辑
        with open(temp_file_path, "rb") as audio:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=audio
            )
        transcript_text = transcription.text

        ai_result = analyze_audio_transcript(transcript_text)
        
        query_vector = get_embedding(ai_result['weakness_search_query'])
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
        
        recommended_teachers = []
        for hit in search_result:
            recommended_teachers.append({
                "bubble_id": hit.payload['bubble_id'],
                "name": hit.payload['name'],
                "match_score": hit.score,
                "specialty": hit.payload['specialty']
            })

        return {
            "transcript": transcript_text,
            "score": ai_result['overall_score'],
            "feedback": ai_result['feedback'],
            "recommendations": recommended_teachers
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
