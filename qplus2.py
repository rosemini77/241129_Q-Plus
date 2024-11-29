import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, auth

# Streamlit secrets에서 Firebase 인증 정보 읽어오기
firebase_config = st.secrets["firebase"]

# Firebase 인증 정보 생성 (서비스 계정 방식)
cred = credentials.Certificate({
    "type": "service_account",
    "project_id": firebase_config["project_id"],
    "private_key_id": firebase_config["private_key_id"],
    "private_key": firebase_config["private_key"].replace("\\n", "\n"),  # 여러 줄을 하나로 합침
    "client_email": firebase_config["client_email"],
    "client_id": firebase_config["client_id"],
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",  # 기본 URL
    "token_uri": "https://oauth2.googleapis.com/token",  # 기본 URL
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",  # 기본 URL
    "client_x509_cert_url": firebase_config["client_x509_cert_url"]
})

# Firebase 초기화
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()  # Firestore 데이터베이스 클라이언트 생성

# Streamlit secrets에서 OpenAI API 키 읽어오기
openai_api_key = st.secrets["openai"]["api_key"]

# PDF 데이터를 Firestore에 저장 (문서 단위로 저장)
def upload_to_firestore_optimized(uploaded_files):
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
            reader = PdfReader(temp_file_path)

            # PDF 전체 데이터를 하나의 Firestore 문서로 저장
            pages_text = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text is None:  # 텍스트 추출 불가 처리
                    text = "[텍스트 추출 불가]"
                pages_text.append({"page_number": page_number, "text": text})

            # Firestore에 PDF 데이터 저장
            document_data = {
                "document_name": uploaded_file.name,
                "pages": pages_text,
            }
            db.collection("pdf_documents").document(uploaded_file.name).set(document_data)

    st.success("PDF 데이터를 Firestore에 저장 완료!")

# Firestore에서 데이터 가져오기 (문서 단위)
def fetch_from_firestore_optimized():
    docs = db.collection("pdf_documents").stream()
    texts = []
    metadatas = []
    for doc in docs:
        data = doc.to_dict()
        document_name = data["document_name"]
        for page in data["pages"]:
            texts.append(page["text"])
            metadatas.append({"document_name": document_name, "page_number": page["page_number"]})
    return texts, metadatas

# Firestore 데이터를 인덱싱
def create_vector_store_from_firestore_optimized():
    texts, metadatas = fetch_from_firestore_optimized()

    # 텍스트를 작은 덩어리로 나누기
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents(texts, metadatas=metadatas)

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY를 설정하세요.")  # API 키 확인

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # API 키 전달
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# QA 체인 생성 함수
def create_qa_chain(vector_store):
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY를 설정하세요.")  # API 키 확인
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=openai_api_key
    )
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    return qa_chain

# 사용자 로그인 확인 함수
def authenticate_user(email, password):
    try:
        # 비밀번호를 Firebase Authentication에서 검증하려면 클라이언트 SDK 사용이 필요하지만,
        # 여기서는 이메일과 비밀번호가 하드코딩된 예시입니다.
        if email.strip() == "dawon2024" and password.strip() == "dawon2024":
            return True
        else:
            return False
    except Exception as e:
        return False

# Streamlit 애플리케이션 메인 함수
def main():
    st.title("PDF 질문 답변 챗봇")

    # 로그인 화면
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        email = st.text_input("아이디", value="dawon2024")
        password = st.text_input("비밀번호", type="password")
        
        if st.button("로그인"):
            if authenticate_user(email, password):
                # 로그인 성공 시 로그인 상태 갱신
                st.session_state.logged_in = True
                with st.spinner("로그인 중..."):
                    st.session_state.logged_in = True  # 로그인 상태 갱신
                    st.rerun()  # 페이지 새로고침으로 다음 화면으로 넘어감
            else:
                st.error("아이디 또는 비밀번호가 잘못되었습니다.")
        return

    # 로그인 후 PDF 파일 업로드 기능
    uploaded_files = st.file_uploader("PDF 파일을 업로드하세요", accept_multiple_files=True, type="pdf")

    if uploaded_files:
        # PDF 파일 데이터를 Firestore에 저장 (문서 단위)
        upload_to_firestore_optimized(uploaded_files)

    # Firestore 데이터로부터 QA 체인 생성
    vector_store = create_vector_store_from_firestore_optimized()
    qa_chain = create_qa_chain(vector_store)

    # 질문 입력 받기
    query = st.text_input("질문을 입력하세요:")

    if query:
        # 질문에 대한 답변 실행
        response = qa_chain.run(query)
        st.write("답변:", response)

        # Firestore 데이터를 활용해 관련 문서와 페이지 번호 출력
        st.write("관련된 문서 및 페이지 번호: ")
        results = vector_store.similarity_search_with_score(query, k=3)  # 상위 3개의 관련 문서 검색
        for result in results:
            metadata = result[0].metadata  # 메타데이터 추출
            st.write(f"문서명: {metadata['document_name']}, 페이지 번호: {metadata['page_number']}")

if __name__ == "__main__":
    main()
