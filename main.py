import os
import json
import shutil
import tempfile
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. 初始化配置
load_dotenv() 

# ⚠️ 部署时，请在 Render/Railway 的 Environment Variables 里填入这些值，不要写死在代码里
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL") 
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 定义向量库集合名称
COLLECTION_NAME = "teachers_skills"

app = FastAPI(title="PandaFreeAI Engine")

# 2. 配置跨域，允许你的 Bubble 域名访问
origins = [
    "http://pandafreeai.com",
    "https://pandafreeai.com",
    "http://version-test.pandafreeai.com", # Bubble 的测试环境域名通常长这样，建议加上
    "*" # 开发阶段为了方便可以全开，上线建议限制
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 启动时检查/创建数据库 ---
@app.on_event("startup")
def startup_event():
    # 检查集合是否存在，不存在则创建
    if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE), # OpenAI embedding 维度是 1536
        )
        print(f"Collection {COLLECTION_NAME} created.")

# --- 核心工具函数 ---

def get_embedding(text: str):
    """调用 OpenAI 将文本转为向量"""
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def analyze_audio_transcript(transcript: str):
    """GPT-4o 评分并提取弱点"""
    system_prompt = """
    You are an expert IELTS Speaking examiner. Analyze the transcript.
    Return JSON with:
    - 'overall_score': Float (0-9)
    - 'feedback': String (Constructive feedback)
    - 'weakness_search_query': String (Describe the core problem to search for a teacher. E.g., 'Teacher specialized in correcting flat intonation and pause fillers')
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

# [后台功能] 添加老师到向量库
# 你可以用 Postman 调用这个接口，把老师数据灌进去
@app.post("/admin/add-teacher")
async def add_teacher(name: str, specialty_desc: str, bubble_id: str, secret_key: str = Header(None)):
    """
    specialty_desc: 详细描述老师擅长什么 (这段文字会被变成向量)
    bubble_id: Bubble 数据库里该老师的 Unique ID
    """
    # 简单加一个密码防止被滥用
    if secret_key != os.getenv("ADMIN_SECRET", "123456"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 1. 计算向量
    vector = get_embedding(specialty_desc)
    
    # 2. 存入 Qdrant
    operation_info = qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()), # 向量库内部ID
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

# [前台功能] 核心业务：评测 + 推荐
@app.post("/assess-audio")
async def assess_audio(file: UploadFile = File(...)):
    # 1. 存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        # 2. Whisper 转录
        with open(temp_file_path, "rb") as audio:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=audio
            )
        transcript_text = transcription.text

        # 3. GPT-4o 诊断
        ai_result = analyze_audio_transcript(transcript_text)
        
        # 4. Qdrant 向量搜索 (RAG)
        # 把用户的弱点描述变成向量
        query_vector = get_embedding(ai_result['weakness_search_query'])
        
        # 在数据库里搜最相似的 3 个老师
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
        
        # 格式化推荐列表给 Bubble
        recommended_teachers = []
        for hit in search_result:
            recommended_teachers.append({
                "bubble_id": hit.payload['bubble_id'], # 拿着这个 ID 去 Bubble 数据库找老师照片
                "name": hit.payload['name'],
                "match_score": hit.score, # 匹配度
                "specialty": hit.payload['specialty']
            })

        return {
            "transcript": transcript_text,
            "score": ai_result['overall_score'],
            "feedback": ai_result['feedback'],
            "recommendations": recommended_teachers
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)