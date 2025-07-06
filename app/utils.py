# utils.py

from langchain.text_splitter import CharacterTextSplitter

def split_text(text: str, chunk_size=500, chunk_overlap=50):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])
