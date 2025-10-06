from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
import re

st.set_page_config(
    page_title="YouTubeGPT",
    page_icon="🎬",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o",
    streaming=True,
    callbacks=[ChatCallbackHandler(),]
)


def extract_video_id(url):
    """YouTube URL에서 비디오 ID 추출"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]*)',
        r'youtube\.com\/embed\/([^&\n?]*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


@st.cache_data(show_spinner="Loading transcript...")
def get_transcript(video_id):
    """YouTube 자막 가져오기"""
    try:
        api = YouTubeTranscriptApi()
        transcripts = api.list(video_id)

        # 한국어 자막 찾기
        try:
            transcript = transcripts.find_transcript(['ko'])
            data = transcript.fetch()
            return " ".join([snippet.text for snippet in data])
        except:
            pass

        # 영어 자막 찾기
        try:
            transcript = transcripts.find_transcript(['en'])
            data = transcript.fetch()
            return " ".join([snippet.text for snippet in data])
        except:
            pass

        # 첫 번째 사용 가능한 자막 가져오기
        for t in transcripts:
            try:
                data = t.fetch()
                return " ".join([snippet.text for snippet in data])
            except:
                continue

        return None
    except Exception as e:
        st.error(f"Error details: {str(e)}")
        return None


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


st.title("YouTubeGPT")

st.markdown("""
Welcome to YouTubeGPT!

Use this tool to generate scripts and summaries from YouTube videos.

Paste a YouTube URL in the sidebar to get started.
""")

with st.sidebar:
    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if youtube_url:
    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.error("Invalid YouTube URL. Please check the link.")
    else:
        transcript = get_transcript(video_id)

        if not transcript:
            st.error("Could not load transcript. The video may not have subtitles.")
        else:
            # 비디오 미리보기
            st.video(youtube_url)

            # 채팅 인터페이스
            if "messages" not in st.session_state:
                st.session_state["messages"] = []

            send_message("I'm ready! Ask me anything about this video!", "ai", save=False)
            paint_history()

            message = st.chat_input("Ask about the video...")
            if message:
                send_message(message, "human")

                prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     """
                     You are a helpful assistant for English learning and video transcript analysis.
                     The user wants to study English through shadowing and understanding video content.

                     When asked for a script or transcript, provide it in full or in organized sections.
                     You can also summarize, explain difficult parts, or answer specific questions.

                     Always be helpful and provide the full transcript when requested.

                     Transcript:
                     {transcript}
                     """),
                    ("human", "{question}"),
                ])

                chain = prompt | llm

                with st.chat_message("ai"):
                    chain.invoke({
                        "transcript": transcript,
                        "question": message
                    })
else:
    st.session_state["messages"] = []
