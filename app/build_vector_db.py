# === 파일 1: build_vector_db.py ===
import os
import time
import chardet
import chromadb
import pandas as pd
import pymupdf4llm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


def detect_encoding(file_path, default='utf-8'):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(4096))
    return result['encoding'] if result['encoding'] else default


def load_csv(file_path):
    encoding = detect_encoding(file_path)
    df = pd.read_csv(file_path, encoding=encoding)
    text_rows = df.astype(str).apply(" | ".join, axis=1).tolist()
    full_text = "\n".join(text_rows)
    print(f"CSV 로드: {os.path.basename(file_path)} (인코딩: {encoding})")
    return full_text


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


if __name__ == "__main__":
    print("[1단계] 문서 로드 및 벡터 DB 구축 시작...")
    csv_dir = r"C:\\Users\\COM\\Desktop\\Botrade\\docs\\data\\csv"
    excel_dir = r"C:\\Users\\COM\\Desktop\\Botrade\\docs\\data\\excel"
    pdf_dir = r"C:\\Users\\COM\\Desktop\\Botrade\\docs\\data\\pdf"

    embedder = SentenceTransformer("intfloat/multilingual-e5-small")
    client = chromadb.PersistentClient(path="./chroma_db")

    collection_name = "rag_collection"
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(name=collection_name)

    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    start = time.time()
    raw_text = load_all_files(csv_dir, excel_dir, pdf_dir)
    chunks = split_text(raw_text)
    embeddings = embedder.encode(chunks, convert_to_tensor=False)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[f"chunk_{i+1}"],
            embeddings=[embedding.tolist()],
            metadatas=[{"text": chunk}]
        )
        if (i+1) % 100 == 0:
            print(f"{i+1}/{len(chunks)}개 저장 완료")

    print(f"[완료] 문서 임베딩 및 저장 완료! 총 소요 시간: {time.time() - start:.2f}초")
