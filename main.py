from fastapi import FastAPI, UploadFile, File, HTTPException
import openai
import json
import os

app = FastAPI()

# 配置你的 API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/api/score_ielts")
async def score_ielts_speaking(file: UploadFile = File(...)):
    try:
        # 1. 保存临时音频文件 (Bubble 传过来的通常是 webm 或 mp3)
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        # 2. STT: 使用 OpenAI Whisper 转录音频
        # 注意：对于口语考试，转录必须精准，Whisper 模型是目前最好的
        with open(temp_filename, "rb") as audio_file:
            transcript_response = openai.Audio.transcribe(
                model="whisper-1", 
                file=audio_file,
                language="en" # 强制识别英语
            )
        
        user_transcript = transcript_response["text"]
        
        # 如果用户没说话或录音为空
        if len(user_transcript) < 10:
            return {"error": "Recording too short or inaudible."}

        # 3. LLM: 使用 GPT-4o 进行评分 (GPT-4o 比 GPT-3.5 更能遵循 JSON 格式)
        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": IELTS_EXAMINER_PROMPT},
                {"role": "user", "content": user_transcript}
            ],
            temperature=0.3, # 降低随机性，让评分更稳定
            response_format={"type": "json_object"} # 强制 JSON 模式 (新版 API 支持)
        )

        # 4. 解析结果
        ai_response_content = completion.choices[0].message.content
        result_json = json.loads(ai_response_content)
        
        # 把转录文本也塞回去，前端可能要展示
        result_json["transcript"] = user_transcript

        # 5. 清理临时文件
        os.remove(temp_filename)

        return result_json

    except Exception as e:
        return {"error": str(e)}