from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)
class ChatCallbackHanddler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *arges, **kwargs):
        self.message_box = st.empty()
    
    def on_llm_end(self, *arges, **kwargs):
        save_message(self.message, "ai")
    
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)



llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o",
    streaming=True,
    callbacks=[ChatCallbackHanddler(),]
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

def save_message(message, role):
        st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join([document.page_content for document in docs])

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
     Answer the question using only th information provided in the document. If you don't know the answer, just say sorry i don't know". Don't try to make up an answer.)

     context: {context}
     """,
    ),
    ("human", "{question}"),
])

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
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),

        } | prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)


else:
    st.session_state["messages"] = []




        