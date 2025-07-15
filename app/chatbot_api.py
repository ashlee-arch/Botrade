from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import retrieve_docs, generate_answer, msgManager, collection, embedder, model

app = FastAPI()

# OpenAI 호환 모델 목록
@app.get("/v1/models")
def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "exaone3.5",
                "object": "model",
                "owned_by": "you"
            }
        ]
    }

# 요청 데이터 모델
class ChatRequest(BaseModel):
    model: str
    messages: list
    stream: bool = False

# Chat Completion API
@app.post("/v1/chat/completions")
def chat_completion(req: ChatRequest):
    try:
        user_query = ""

        # 실제 유저 질문만 추출 (Chatbox 자동 메시지 필터링 포함)
        for msg in reversed(req.messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                content = "".join([c.get("text", "") for c in content if isinstance(c, dict)])
            if "give this conversation a name" in content.lower():
                continue
            user_query = content.strip()
            break

        if not user_query:
            return {"error": "user 메시지가 없습니다."}

        retrieved = retrieve_docs(user_query, collection, embedder, top_k=2)
        answer = generate_answer(user_query, retrieved, model, msgManager)
        msgManager.append_msg_by_assistant(answer)

        # 🔐 str 보장 + 로그 출력
        response = {
            "id": "chatcmpl-001",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(answer)
                    },
                    "finish_reason": "stop"
                }
            ],
            "model": "exaone3.5"
        }
        print("📤 최종 응답 구조:", response)
        return response

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": f"서버 내부 오류: {str(e)}"}
