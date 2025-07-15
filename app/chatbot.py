import time
import ollama
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import deque

# === 메시지 관리 클래스 ===
class Message_manager:
    def __init__(self):
        self._system_msg = {"role": "system", "content": ""}
        self.queue = deque(maxlen=10)

    def create_msg(self, role, content):
        if not isinstance(content, str):
            # 예외적으로 content가 list나 dict이면 문자열화
            if isinstance(content, list):
                content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
            elif isinstance(content, dict):
                import json
                content = json.dumps(content)
            else:
                content = str(content)
        return {"role": role, "content": content}

    def system_msg(self, content):
        self._system_msg = self.create_msg("system", content)

    def append_msg(self, content):
        self.queue.append(self.create_msg("user", content))

    def append_msg_by_assistant(self, content):
        self.queue.append(self.create_msg("assistant", content))

    def generate_prompt(self, retrieved_docs):
        docs = "\n".join(retrieved_docs)
        return [self._system_msg, {
            "role": "system",
            "content": f"참고 문서 내용:\n{docs}\n위 내용을 참고하여 중소기업 수출 담당자가 이해할 수 있도록 친절하고 쉽게 설명해 주세요. 모르는 부분은 '잘 모르겠습니다'라고 답해 주세요."
        }] + list(self.queue)

# === 질의 임베딩 및 문서 검색 ===
def retrieve_docs(query, collection, embedder, top_k=2):
    start = time.time()
    query_embedding = embedder.encode(query, convert_to_tensor=False)

    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    if isinstance(query_embedding[0], list):  # 2D → 1D flatten
        query_embedding = query_embedding[0]

    print(f"임베딩 생성: {time.time() - start:.2f}초")

    start = time.time()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    print(f"벡터 검색: {time.time() - start:.2f}초")

    if not results["metadatas"]:
        return ["관련 문서를 찾을 수 없습니다."]
    return [doc["text"] for doc in results["metadatas"][0]]

# === Ollama 응답 생성 ===
def generate_answer(query, retrieved_docs, model, msgManager):
    msgManager.append_msg(query)
    msg = msgManager.generate_prompt(retrieved_docs)

    print("💬 메시지 구조:")
    for m in msg:
        print(m)

    try:
        # 스트리밍 없이 단건 응답
        response = ollama.chat(model=model, messages=msg, stream=False)
        answer = response["message"]["content"]
        print("✅ 응답:", answer)
        return answer
    except Exception as e:
        print(f"[Ollama 오류] {e}")
        return "답변 생성 실패"
    

# === 인스턴스 초기화 ===
model = "exaone3.5:latest"
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="rag_collection")

msgManager = Message_manager()
msgManager.system_msg(
    "질문자의 말투와 맥락을 고려하여, 이어지는 대화처럼 자연스럽게 응답합니다. "
    "항상 중소기업 수출 담당자가 이해하기 쉽게 설명하고, 필요한 경우 예시를 덧붙입니다. "
    "모르는 경우 '잘 모르겠습니다'라고 답하며, 근거 없는 추측은 하지 않습니다. "
    "대화는 'user'의 마지막 질문에 답하는 형식이며, 이전 대화도 함께 고려하여 응답해야 합니다. "
    "개행은 문장이 끝날 때와 주제가 바뀔 때만 사용하고, 불필요한 줄바꿈은 피합니다."
)

# === 로컬 콘솔 테스트용 대화 루프 ===
def chat_loop(collection, embedder, model):
    local_msgManager = Message_manager()
    local_msgManager.system_msg(
        "질문자의 말투와 맥락을 고려하여, 이어지는 대화처럼 자연스럽게 응답합니다. "
        "항상 중소기업 수출 담당자가 이해하기 쉽게 설명하고, 필요한 경우 예시를 덧붙입니다. "
        "모르는 경우 '잘 모르겠습니다'라고 답하며, 근거 없는 추측은 하지 않습니다. "
        "대화는 'user'의 마지막 질문에 답하는 형식이며, 이전 대화도 함께 고려하여 응답해야 합니다. "
        "개행은 문장이 끝날 때와 주제가 바뀔 때만 사용하고, 불필요한 줄바꿈은 피합니다."
    )

    print("RAG 챗봇 시작! 질문 입력 (종료하려면 'exit'):")
    while True:
        query = input("> ")
        if query.lower() == "exit":
            print("챗봇 종료!")
            break

        start_time = time.time()
        retrieved_docs = retrieve_docs(query, collection, embedder, top_k=2)
        answer = generate_answer(query, retrieved_docs, model, local_msgManager)
        local_msgManager.append_msg_by_assistant(answer)
        print(f"[총 소요 시간: {time.time() - start_time:.2f}초]\n")

# === 직접 실행 시 콘솔 모드 실행 ===
if __name__ == "__main__":
    import numpy as np

    model = "exaone3.5:latest"
    embedder = SentenceTransformer("intfloat/multilingual-e5-small")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="rag_collection")

    msgManager = Message_manager()
    msgManager.system_msg(
        "질문자의 말투와 맥락을 고려하여, 이어지는 대화처럼 자연스럽게 응답합니다. "
        "항상 중소기업 수출 담당자가 이해하기 쉽게 설명하고, 필요한 경우 예시를 덧붙입니다. "
        "모르는 경우 '잘 모르겠습니다'라고 답하며, 근거 없는 추측은 하지 않습니다. "
        "대화는 'user'의 마지막 질문에 답하는 형식이며, 이전 대화도 함께 고려하여 응답해야 합니다. "
        "개행은 문장이 끝날 때와 주제가 바뀔 때만 사용하고, 불필요한 줄바꿈은 피합니다."
    )

    print("RAG 챗봇 시작! 질문 입력 (종료하려면 'exit'):")
    while True:
        query = input("> ")
        if query.lower() == "exit":
            print("챗봇 종료!")
            break

        start = time.time()
        query_embedding = embedder.encode(query, convert_to_tensor=False)
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        if isinstance(query_embedding[0], list):  # 2D → 1D
            query_embedding = query_embedding[0]
        print(f"임베딩 생성: {time.time() - start:.2f}초", end=' | ')

        start = time.time()
        results = collection.query(query_embeddings=[query_embedding], n_results=2)
        print(f"벡터 검색: {time.time() - start:.2f}초")
        if not results["metadatas"]:
            retrieved = ["관련 문서를 찾을 수 없습니다."]
        else:
            retrieved = [doc["text"] for doc in results["metadatas"][0]]

        answer = generate_answer(query, retrieved, model, msgManager)
        msgManager.append_msg_by_assistant(answer)
        print(f"답변: {answer}\n")