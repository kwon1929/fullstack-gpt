import time
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)



@st.cache_resource(show_spinner=True)
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    # 3. 문서 로딩 및 분할
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    # 4. 임베딩 + 캐시 
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings,cache_dir)
    # 5. 벡터스토어 저장 및 리트리버 설정 
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save: 
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


st.title("DocumentGPT")

st.markdown("""
Welcome to DocumentGPT! 
            
Use this DocumentGPT to generate summaries, insights, and analyses from your documents.
            
you can upload a file in sidebar
""")

with st.sidebar:
    file = st.file_uploader("Upload your document", type=["pdf", "docx", "txt"])

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask me anything!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask a question about the document...")
    if message:
        send_message(message, "human")

else:
    st.session_state["messages"] = []




        