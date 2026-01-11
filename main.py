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

# 1. 修改分析函数，增加 question 参数
def analyze_audio_transcript(transcript: str, question: str):
    # 在 Prompt 中明确告诉 AI 题目是什么
    system_prompt = f"""
    You are an expert IELTS Speaking examiner. 
    The student is answering the following question: "{question}"
    
    Analyze the transcript based on:
    1. Fluency and Coherence
    2. Lexical Resource
    3. Grammatical Range and Accuracy
    4. Pronunciation
    5. Task Response (Did they answer the specific question?)

    Return JSON with: 'overall_score', 'feedback', 'improvement_suggestions', 'weakness_search_query'.
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
async def assess_audio(
    # 注意：这里类型提示允许 UploadFile 或 str
    file: Union[UploadFile, str] = File(...), 
    question_text: str = Form(...)
):
    # 1. 定义临时文件路径 (默认使用 webm，兼容性最好)
    temp_filename = f"temp_{uuid.uuid4()}.webm"
    
    try:
        # 2. 【核心修复】判断文件来源是“上传”还是“链接”
        if isinstance(file, str):
            # 情况 A: Bubble 传过来的是 URL 字符串 (最常见)
            print(f"📥 Downloading file from URL: {file[:50]}...")
            urllib.request.urlretrieve(file, temp_filename)
        else:
            # 情况 B: Bubble 传过来的是二进制文件对象
            print(f"📥 Receiving binary file: {file.filename}")
            # 如果原始文件有后缀，尽量保留原始后缀
            if file.filename and "." in file.filename:
                ext = file.filename.split(".")[-1]
                temp_filename = f"temp_{uuid.uuid4()}.{ext}"
            
            content = await file.read()
            with open(temp_filename, "wb") as f:
                f.write(content)

        # 检查文件大小，防止空文件报错
        if os.path.getsize(temp_filename) == 0:
            raise Exception("Received file is empty (0 bytes).")

        # 3. Whisper 转录 (OpenAI)
        print("🎙️ Sending to Whisper...")
        with open(temp_filename, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        transcript_text = transcription.text
        print(f"📝 Transcript: {transcript_text[:50]}...")

        # 4. 调用分析函数 (GPT-4o)
        print(f"🧠 Analyzing answer for: {question_text}")
        ai_result = analyze_audio_transcript(transcript_text, question_text)
        
        # 5. Qdrant 向量搜索 (搜索老师)
        recommended_teachers = []
        if qdrant:
            try:
                # 只有当 AI 成功返回了 weakness_search_query 才去搜
                search_query = ai_result.get('weakness_search_query', 'IELTS speaking teacher')
                print(f"🔍 Searching teachers for: {search_query}")
                
                query_vector = get_embedding(search_query)
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
                print(f"⚠️ Search warning: {e}")
                # 搜索出错不影响主流程，给个空列表
                recommended_teachers = []

        # 6. 返回结果
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
        print(f"❌ Error in assess_audio: {str(e)}")
        # 打印详细错误方便调试
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
        
    finally:
        # 清理垃圾文件
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

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
