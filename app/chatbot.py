import os
import time
import ollama
import chardet
import chromadb
import pymupdf4llm
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from collections import deque

# 사용 모델
model = "exaone3.5:latest"

# === 1. 파일 처리 함수 === #


def detect_encoding(file_path, default='utf-8'):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(4096))  # 일부만 검사
    return result['encoding'] if result['encoding'] else default


def load_csv(file_path):
    try:
        # 인코딩 자동 감지
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)
        text_rows = df.astype(str).apply(" | ".join, axis=1).tolist()
        full_text = "\n".join(text_rows)
        print(f"CSV 로드: {os.path.basename(file_path)} (인코딩: {encoding})")
        return full_text
    except Exception as e:
        raise RuntimeError(f"{os.path.basename(file_path)} 읽기 실패: {e}")

def load_excel(file_path):
    df = pd.read_excel(file_path)
    text_rows = df.astype(str).apply(" | ".join, axis=1).tolist()
    full_text = "\n".join(text_rows)
    print(f"Excel 로드: {os.path.basename(file_path)}")
    return full_text

def load_pdf(file_path):
    pdf_data = pymupdf4llm.to_markdown(file_path)
    text = "".join(pdf_data)
    print(f"PDF 로드: {os.path.basename(file_path)}")
    return text

# === 2. 폴더 내 모든 파일 로딩 === #

def load_all_files(csv_dir, excel_dir, pdf_dir):
    all_text = []

    def load_folder(folder_path, loader_func, ext):
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(ext):
                fpath = os.path.join(folder_path, fname)
                try:
                    all_text.append(loader_func(fpath))
                except Exception as e:
                    print(f"[오류] {fname}: {e}")

    load_folder(csv_dir, load_csv, ".csv")
    load_folder(excel_dir, load_excel, ".xlsx")
    load_folder(pdf_dir, load_pdf, ".pdf")

    return "\n".join(all_text)

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

# === 3. 메시지 관리 클래스 === #

class Message_manager:
    def __init__(self):
        self._system_msg = {"role": "system", "content": ""}
        self.queue = deque(maxlen=10)

    def create_msg(self, role, content):
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
            "content": f"문서 내용: {docs}\n질문에 대한 답변은 문서 내용을 기반으로 정확히 제공하시오."
        }] + list(self.queue)

# === 4. 질문 처리 함수 === #

def retrieve_docs(query, collection, embedder, top_k=2):
    start = time.time()
    query_embedding = embedder.encode(query, convert_to_tensor=False)
    print(f"임베딩 생성: {time.time() - start:.2f}초")

    start = time.time()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    print(f"벡터 검색: {time.time() - start:.2f}초")

    if not results["metadatas"]:
        return ["관련 문서를 찾을 수 없습니다."]
    return [doc["text"] for doc in results["metadatas"][0]]

# === 5. 답변 생성 함수 === #

def generate_answer(query, retrieved_docs):
    msgManager.append_msg(query)
    msg = msgManager.generate_prompt(retrieved_docs)

    print("답변: ", end="", flush=True)
    full_answer = ""
    start = time.time()

    try:
        for response in ollama.chat(model=model, messages=msg, stream=True):
            chunk = response["message"]["content"]
            print(chunk, end="", flush=True)
            full_answer += chunk
    except Exception as e:
        print(f"\n[Ollama 오류] {e}")
        return "답변 생성 실패"

    print()
    print(f"LLM 추론: {time.time() - start:.2f}초")
    return full_answer

# === 6. 대화 루프 === #

def chat_loop():
    print("RAG 챗봇 시작! 질문 입력 (종료하려면 'exit'):")

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

# === 메인 실행 === #

if __name__ == "__main__":
    print("환경 구성 중...")

    # 문서 폴더 경로
    csv_dir = r"C:\Users\COM\Desktop\Botrade\docs\data\csv"
    excel_dir =r"C:\Users\COM\Desktop\Botrade\docs\data\excel"
    pdf_dir = r"C:\Users\COM\Desktop\Botrade\docs\data\pdf"

    # 임베딩 모델 및 벡터DB 초기화
    embedder = SentenceTransformer("intfloat/multilingual-e5-small")
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("rag_collection")
    except Exception:
        pass

    collection = client.get_or_create_collection(name="rag_collection", metadata={"hnsw:space": "cosine"})

    print("문서 로드 및 임베딩 시작...")
    start_time = time.time()

    raw_text = load_all_files(csv_dir, excel_dir, pdf_dir)
    chunks = split_text(raw_text)
    embeddings = embedder.encode(chunks, convert_to_tensor=False)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[f"chunk_{i+1}"],
            embeddings=[embedding.tolist()],
            metadatas=[{"text": chunk}],
        )

    print(f"문서 준비 완료! 소요 시간: {time.time() - start_time:.2f}초")

    msgManager = Message_manager()
    msgManager.system_msg(
        "가장 마지막 'user'의 'content'에 대해 답변한다. "
        "질문에 답할 때는 'system' 메시지 중 '문서 내용'에 명시된 부분을 우선 참고하여 정확히 답한다. "
        "개행은 문장이 끝날 때와 서로 다른 주제나 항목을 구분할 때 사용하며, 불필요한 개행은 넣지 않는다."
    )

    chat_loop()
