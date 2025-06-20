from app.rag_pipeline import retrieve_and_generate

def handle_user_query(question: str) -> str:
    """
    사용자 질문을 받아서 적절한 응답 생성
    """
    response = retrieve_and_generate(question)
    return response
