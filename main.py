import os
import json
import shutil
import tempfile
import uuid
import urllib.request
import requests
from typing import List, Union

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from zhipuai import ZhipuAI

# ================= 1. 配置与初始化 =================
load_dotenv()

# --- OpenAI & 雅思配置 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
COLLECTION_NAME_IELTS = "teachers_skills" # 雅思搜索用的旧表

# --- 智谱AI & 老师配置 (保留你原来的硬编码配置) ---
ZHIPU_API_KEY = "1d2423311eb947bab8f94d3c93c2c3f4.6tKAtiorGYhY7zxh"
QDRANT_URL = "https://fbe88bb7-d113-491f-a57c-0d13aeb30fdb.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY_ZHIPU = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.AZl1rnrCRxr8VUOYlf4-PXnxjVJbDtrm_fFYa6D-0kE"
COLLECTION_NAME_ZHIPU = "tutors_zhipu_v1" # 智谱老师表

zhipu_client = ZhipuAI(api_key=ZHIPU_API_KEY)

# --- Qdrant 客户端 ---
qdrant = None
try:
    print(f"Connecting to Qdrant...")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY_ZHIPU)
    print("Qdrant Client initialized.")
except Exception as e:
    print(f"⚠️ Warning: Qdrant connection failed: {e}")

app = FastAPI(title="PandaFreeAI Engine (Unified)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 2. 数据模型 =================
class TutorSyncRequest(BaseModel):
    bubble_id: str
    name: str
    price: float
    tags: List[str]
    description: str

class MatchRequest(BaseModel):
    user_tags: List[str]
    max_price: float

# ================= 3. 辅助函数 =================

# --- OpenAI 相关函数 ---
def get_openai_embedding(text: str):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def analyze_audio_transcript(transcript: str, question: str):
    system_prompt = f"""
    You are an expert IELTS Speaking examiner. 
    The student is answering the following question: "{question}"
    
    Analyze the transcript based on:
    1. Fluency and Coherence
    2. Lexical Resource
    3. Grammatical Range and Accuracy
    4. Pronunciation
    5. Task Response

    Return JSON with: 'overall_score', 'short_evaluation', 'detailed_feedback', 'improvement_suggestions', 'weakness_search_query'.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": transcript}]
    )
    return json.loads(response.choices[0].message.content)

# --- 智谱相关函数 ---
def get_deterministic_uuid(bubble_id: str) -> str:
    NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
    return str(uuid.uuid5(NAMESPACE, bubble_id))

def get_zhipu_embedding(text: str) -> List[float]:
    response = zhipu_client.embeddings.create(model="embedding-2", input=text)
    return response.data[0].embedding

# ================= 4. 启动事件 =================
@app.on_event("startup")
def startup_event():
    """启动时检查并创建智谱的老师向量表"""
    try:
        if qdrant and not qdrant.collection_exists(COLLECTION_NAME_ZHIPU):
            print(f"正在创建集合: {COLLECTION_NAME_ZHIPU} ...")
            qdrant.recreate_collection(
                collection_name=COLLECTION_NAME_ZHIPU,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            )
            qdrant.create_payload_index(COLLECTION_NAME_ZHIPU, "price", models.PayloadSchemaType.INTEGER)
            qdrant.create_payload_index(COLLECTION_NAME_ZHIPU, "bubble_id", models.PayloadSchemaType.KEYWORD)
            print("✅ 数据库初始化完成！")
        else:
            print("✅ 数据库已存在，准备就绪。")
    except Exception as e:
        print(f"初始化警告 (可忽略): {e}")

# ================= 5. 核心接口 =================

# 🎯 接口 1: 雅思语音评测 (OpenAI)
@app.post("/assess-audio")
async def assess_audio(file: Union[UploadFile, str] = File(...), question_text: str = Form(...)):
    temp_filename = f"temp_{uuid.uuid4()}.webm"
    try:
        if isinstance(file, str):
            req = urllib.request.Request(file, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(temp_filename, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        else:
            if file.filename and "." in file.filename:
                temp_filename = f"temp_{uuid.uuid4()}.{file.filename.split('.')[-1]}"
            content = await file.read()
            with open(temp_filename, "wb") as f:
                f.write(content)

        if os.path.getsize(temp_filename) == 0: raise Exception("Received file is empty.")

        with open(temp_filename, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        transcript_text = transcription.text

        ai_result = analyze_audio_transcript(transcript_text, question_text)
        
        # 搜索老师 (使用 OpenAI 向量搜索旧表)
        recommended_teachers = []
        if qdrant:
            try:
                search_query = ai_result.get('weakness_search_query', 'IELTS speaking teacher')
                query_vector = get_openai_embedding(search_query)
                search_result = qdrant.search(
                    collection_name=COLLECTION_NAME_IELTS,
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
                print(f"⚠️ Search warning: {e}")

        return {
            "status": "success",
            "transcript": transcript_text,
            "overall_score": ai_result.get('overall_score'),
            "short_evaluation": ai_result.get('short_evaluation'),
            "detailed_feedback": ai_result.get('detailed_feedback'),
            "improvement_suggestions": ai_result.get('improvement_suggestions'),
            "recommendations": recommended_teachers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

# 🎯 接口 2: 同步外教数据 (智谱AI)
@app.post("/sync_tutor")
def sync_tutor(tutor: TutorSyncRequest):
    try:
        text_to_embed = f"{tutor.name}。擅长：{' '.join(tutor.tags)}。简介：{tutor.description}"
        vector = get_zhipu_embedding(text_to_embed)
        
        qdrant.upsert(
            collection_name=COLLECTION_NAME_ZHIPU,
            points=[
                models.PointStruct(
                    id=get_deterministic_uuid(tutor.bubble_id),
                    vector=vector,
                    payload={
                        "bubble_id": tutor.bubble_id, "name": tutor.name,
                        "price": int(tutor.price), "tags": tutor.tags, "full_info": text_to_embed
                    }
                )
            ]
        )
        return {"status": "success", "msg": f"老师 {tutor.name} 已同步"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🎯 接口 3: 智能匹配老师 (智谱AI)
@app.post("/recommend")
def recommend(req: MatchRequest):
    try:
        query_text = f"寻找老师，需求：{' '.join(req.user_tags)}"
        query_vector = get_zhipu_embedding(query_text)
        
        search_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME_ZHIPU}/points/search"
        search_payload = {
            "vector": query_vector, "limit": 3, "with_payload": True,
            "filter": {"must": [{"key": "price", "range": {"lte": int(req.max_price)}}]}
        }
        res = requests.post(search_url, headers={"api-key": QDRANT_API_KEY_ZHIPU}, json=search_payload)
        search_result = res.json().get("result", [])
        
        if not search_result: return []

        context_list = [f"- ID: {hit['payload']['bubble_id']}, 姓名: {hit['payload']['name']}, 资料: {hit['payload']['full_info']}" for hit in search_result]
        
        prompt = f"""你是一个课程顾问。请根据用户需求和以下候选人资料，推荐这几位老师。
        用户需求：{req.user_tags} \n候选人列表：\n{chr(10).join(context_list)}
        请务必只返回一个纯JSON数组：[{{"bubble_id": "...", "reason": "30字以内的推荐理由"}}]"""

        response = zhipu_client.chat.completions.create(model="glm-4-flash", messages=[{"role": "user", "content": prompt}], temperature=0.7)
        content = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
