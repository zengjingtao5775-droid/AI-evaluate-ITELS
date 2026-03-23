import os
import json
import shutil
import tempfile
import uuid
import urllib.request
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

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
COLLECTION_NAME_IELTS = "teachers_skills" 

ZHIPU_API_KEY = "1d2423311eb947bab8f94d3c93c2c3f4.6tKAtiorGYhY7zxh"
COLLECTION_NAME_ZHIPU = "tutors_zhipu_v1" 
zhipu_client = ZhipuAI(api_key=ZHIPU_API_KEY)

qdrant = None
try:
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
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

def get_deterministic_uuid(bubble_id: str) -> str:
    NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
    return str(uuid.uuid5(NAMESPACE, bubble_id))

def get_zhipu_embedding(text: str) -> List[float]:
    response = zhipu_client.embeddings.create(model="embedding-2", input=text)
    return response.data[0].embedding

# ================= 4. 启动事件 (已修复兼容性) =================
@app.on_event("startup")
def startup_event():
    """启动时检查并自动创建所需的两张表（兼容所有 Qdrant 版本）"""
    if not qdrant:
        return
    try:
        # 获取现有的所有表名
        existing_collections = [c.name for c in qdrant.get_collections().collections]

        if COLLECTION_NAME_IELTS not in existing_collections:
            print(f"正在创建雅思新表: {COLLECTION_NAME_IELTS} ...")
            qdrant.recreate_collection(
                collection_name=COLLECTION_NAME_IELTS,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            )
            qdrant.create_payload_index(COLLECTION_NAME_IELTS, "bubble_id", models.PayloadSchemaType.KEYWORD)

        if COLLECTION_NAME_ZHIPU not in existing_collections:
            print(f"正在创建智谱新表: {COLLECTION_NAME_ZHIPU} ...")
            qdrant.recreate_collection(
                collection_name=COLLECTION_NAME_ZHIPU,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            )
            qdrant.create_payload_index(COLLECTION_NAME_ZHIPU, "price", models.PayloadSchemaType.INTEGER)
            qdrant.create_payload_index(COLLECTION_NAME_ZHIPU, "bubble_id", models.PayloadSchemaType.KEYWORD)

    except Exception as e:
        print(f"初始化数据库表失败: {e}")

# ================= 5. 核心接口 =================

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

@app.post("/recommend")
def recommend(req: MatchRequest):
    try:
        query_text = f"寻找老师，需求：{' '.join(req.user_tags)}"
        query_vector = get_zhipu_embedding(query_text)
        
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME_ZHIPU,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="price",
                        range=models.Range(lte=int(req.max_price))
                    )
                ]
            ),
            limit=3
        )
        
        if not search_result: return []

        context_list = [f"- ID: {hit.payload['bubble_id']}, 姓名: {hit.payload['name']}, 资料: {hit.payload['full_info']}" for hit in search_result]
        
        prompt = f"""你是一个课程顾问。请根据用户需求和以下候选人资料，推荐这几位老师。
        用户需求：{req.user_tags} \n候选人列表：\n{chr(10).join(context_list)}
        请务必只返回一个纯JSON数组：[{{"bubble_id": "...", "reason": "30字以内的推荐理由"}}]"""

        response = zhipu_client.chat.completions.create(model="glm-4-flash", messages=[{"role": "user", "content": prompt}], temperature=0.7)
        content = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================= 6. 强制建表专属通道 (已修复兼容性) =================
@app.get("/init-db")
def init_database():
    """在浏览器访问这个地址，强制初始化两张表"""
    if not qdrant:
        return {"error": "Qdrant 客户端未连接，请检查环境变量"}
    
    results = []
    try:
        existing_collections = [c.name for c in qdrant.get_collections().collections]

        if COLLECTION_NAME_IELTS not in existing_collections:
            qdrant.recreate_collection(
                collection_name=COLLECTION_NAME_IELTS,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            )
            qdrant.create_payload_index(COLLECTION_NAME_IELTS, "bubble_id", models.PayloadSchemaType.KEYWORD)
            results.append(f"✅ 雅思表 {COLLECTION_NAME_IELTS} 创建成功！")
        else:
            results.append(f"⚡ 雅思表 {COLLECTION_NAME_IELTS} 已经存在了。")

        if COLLECTION_NAME_ZHIPU not in existing_collections:
            qdrant.recreate_collection(
                collection_name=COLLECTION_NAME_ZHIPU,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            )
            qdrant.create_payload_index(COLLECTION_NAME_ZHIPU, "price", models.PayloadSchemaType.INTEGER)
            qdrant.create_payload_index(COLLECTION_NAME_ZHIPU, "bubble_id", models.PayloadSchemaType.KEYWORD)
            results.append(f"✅ 智谱表 {COLLECTION_NAME_ZHIPU} 创建成功！")
        else:
            results.append(f"⚡ 智谱表 {COLLECTION_NAME_ZHIPU} 已经存在了。")

        return {"status": "success", "details": results}

    except Exception as e:
        return {"status": "error", "message": f"建表失败: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
