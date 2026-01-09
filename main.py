import os
import json
import shutil
import tempfile
import uuid
import urllib.request
import qdrant_client # 导入库以便检查版本
from typing import List, Union
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- 1. 初始化配置 ---
load_dotenv()

# 从环境变量获取 Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "123456")

client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION_NAME = "teachers_skills"

app = FastAPI(title="PandaFreeAI Engine (Debug Mode)")

# 打印当前安装的 Qdrant 版本到日志
print(f"🔍 DEBUG INFO: Installed qdrant-client version: {qdrant_client.__version__}")
print(f"🔍 DEBUG INFO: Qdrant Object attributes: {dir(qdrant)}")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. 启动事件 ---
@app.on_event("startup")
def startup_event():
    # 为了防止启动报错，这里加个 try
    try:
        if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            print(f"Collection {COLLECTION_NAME} created.")
    except Exception as e:
        print(f"⚠️ Startup Warning: {str(e)}")

# --- 3. 工具函数 ---
def get_embedding(text: str):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def analyze_audio_transcript(transcript: str):
    system_prompt = """
    You are an expert IELTS Speaking examiner. Analyze the transcript.
    Return a strictly valid JSON object with:
    - 'overall_score': Float (0-9)
    - 'feedback': String (Constructive feedback)
    - 'weakness_search_query': String (Core weakness for search)
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

# --- 4. API 接口 ---

@app.post("/assess-audio")
async def assess_audio(file: Union[UploadFile, str] = File(...)):
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        # 1. 下载音频
        if isinstance(file, str):
            req = urllib.request.Request(file, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(temp_file_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        else:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # 2. 转录
        with open(temp_file_path, "rb") as audio:
            transcription = client.audio.transcriptions.create(model="whisper-1", file=audio)
        transcript_text = transcription.text

        # 3. 评分
        ai_result = analyze_audio_transcript(transcript_text)
        
        # 4. RAG 搜索 (关键报错点)
        query_vector = get_embedding(ai_result['weakness_search_query'])
        
        recommended_teachers = []
        
        # 🛡️ 调试代码：检查 search 方法是否存在
        if hasattr(qdrant, 'search'):
            # 如果 search 存在，正常执行
            search_result = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=3
            )
            for hit in search_result:
                recommended_teachers.append({
                    "bubble_id": hit.payload['bubble_id'],
                    "name": hit.payload['name'],
                    "match_score": hit.score,
                    "specialty": hit.payload['specialty']
                })
        else:
            # 🚨 如果 search 不存在，把版本信息作为“推荐结果”返回，方便我们在 Bubble 里看到
            print("🚨 ERROR: 'search' method missing!")
            # 尝试使用旧版方法 (兼容性处理)
            try:
                # 这是一个旧版本的写法，试试看能不能碰运气
                search_result = qdrant.search_groups(
                    collection_name=COLLECTION_NAME,
                    vector=query_vector,
                    group_by="bubble_id",
                    limit=3
                )
            except:
                # 如果还是不行，返回调试信息
                recommended_teachers.append({
                    "bubble_id": "DEBUG_ERROR",
                    "name": f"Ver: {qdrant_client.__version__}",
                    "match_score": 0,
                    "specialty": "Please check Render logs or requirements.txt"
                })

        return {
            "status": "success",
            "transcript": transcript_text,
            "score": ai_result['overall_score'],
            "feedback": ai_result['feedback'],
            "recommendations": recommended_teachers
        }

    except Exception as e:
        # 将具体错误返回给 Bubble，而不是笼統的 500
        raise HTTPException(status_code=500, detail=f"Debug Error: {str(e)} | Version: {qdrant_client.__version__}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Admin 接口保持不变 (略)
@app.post("/admin/add-teacher")
async def add_teacher(name: str = Form(...), specialty_desc: str = Form(...), bubble_id: str = Form(...), secret_key: str = Header(None)):
    if secret_key != ADMIN_SECRET: raise HTTPException(status_code=401)
    vector = get_embedding(specialty_desc)
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"bubble_id": bubble_id, "name": name, "specialty": specialty_desc})])
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
