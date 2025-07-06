import pymupdf4llm

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import ollama
import time
from collections import deque

file_path = "{파일경로+파일명}"  # 실제 PDF
model = "exaone3.5:latest"  # 사용할 Ollama 모델

# 1. PDF 로드 및 청킹
def load_pdf(file_path):

    pdf_data = pymupdf4llm.to_markdown(file_path)
    text = "".join(pdf_data)  # 페이지별 텍스트를 하나의 문자열로 결합
    print("PDF파일 로드..")
    return text


def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)


# 2. 임베딩 및 벡터 DB 저장

print("환경 구성 중.")
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
client = chromadb.PersistentClient(path="./chroma_db")
client.delete_collection("rag_collection")
collection = client.get_or_create_collection(name="rag_collection",  metadata={"hnsw:space": "cosine"} ) # 코사인 메트릭 설정)

print("문서 로드 및 임베딩 시작...")
start_time = time.time()
raw_text = load_pdf(file_path)
chunks = split_text(raw_text)
embeddings = embedder.encode(chunks, convert_to_tensor=False)

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    collection.add(
        ids=[f"chunk_{i + 1 }"],
        embeddings=[embedding.tolist()],
        metadatas=[{"text": chunk}],
    )
    
print(f"문서 준비 완료! 소요 시간: {time.time() - start_time:.2f}초")

# 3. 질문 처리 함수
def retrieve_docs(query, collection, embedder, top_k=2):
    start = time.time()
    query_embedding = embedder.encode(query, convert_to_tensor=False)

    print(f"임베딩 생성: {time.time() - start:.2f}초")

    start = time.time()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    print(f"벡터 검색: {time.time() - start:.2f}초")

    if not results["metadatas"]:  # 검색 결과가 없는 경우
        return ["관련 문서를 찾을 수 없습니다."]

    docs = [doc["text"] for doc in results["metadatas"][0]]

    return docs

# 4. 메세지 매니저 클래스 구현
class Message_manager:
    def __init__(self):
        self._system_msg = {"role": "system", "content": ""}
        self.queue = deque(maxlen=10)  # 최대 10개 대화 저장

    def create_msg(self, role, content):
        return {"role": role, "content": content}

    def system_msg(self, content):
        self._system_msg = self.create_msg("system", content)

    def append_msg(self, content):
        msg = self.create_msg("user", content)
        self.queue.append(msg)

    def get_chat(self):
        return [self._system_msg] + list(self.queue)

    def set_retrived_docs(self, docs):
        self.retrieved_docs = docs

    def append_msg_by_assistant(self, content):
        msg = self.create_msg("assistant", content)
        self.queue.append(msg)

    def generate_prompt(self, retrieved_docs):

        docs = "\n".join(retrieved_docs)

        prompt = [msgManager._system_msg,{
            "role": "system",
            "content": f"문서 내용: {docs}\n질문에 대한 답변은 문서 내용을 기반으로 정확히 제공하시오.",
        }] + list(msgManager.queue)

        return prompt


msgManager = Message_manager()


# 시스템 메세지 등록
msgManager.system_msg(
    "가장 마지막 'user'의 'content'에 대해 답변한다."
    "질문에 답할 때는 'system' 메시지 중 '문서 내용'에 명시된 부분을 우선 참고하여 정확히 답한다."
    "개행은 문장이 끝날때와 서로 다른 주제나 항목을 구분할 때 사용하며, 불필요한 개행은 넣지 않는다."
)

# 5. 답변 생성 함수
def generate_answer(query, retrieved_docs):
    start = time.time()

    # 이전 대화 기록 추가 (최대 10개)
    msgManager.append_msg(query)

    # 스트리밍으로 답변 생성
    print("답변: ", end="", flush=True)
    full_answer = ""

    msg = msgManager.generate_prompt(retrieved_docs)

    for response in ollama.chat(model=model, messages=msg, stream=True):
        chunk = response["message"]["content"]
        print(chunk, end="", flush=True)  # 한 글자씩 출력
        full_answer += chunk
    print()  # 줄바꿈
    print(f"LLM 추론: {time.time() - start:.2f}초")
    return full_answer

# 6. 대화형 루프
def chat_loop():

    print("RAG 챗봇 시작! 질문 입력 (종료하려면 'exit' 입력):")

    while True:
        query = input("> ")
        if query.lower() == "exit":
            print("챗봇 종료!")
            break

        start_time = time.time()
        retrieved_docs = retrieve_docs(query, collection, embedder, top_k=2)

        answer = generate_answer(query, retrieved_docs)
        msgManager.append_msg_by_assistant(answer)

        print(f"[총 소요 시간: {time.time() - start_time:.2f}초]\n")


# 실행
if __name__ == "__main__":
    chat_loop()