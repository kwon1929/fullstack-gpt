from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
import streamlit as st
import os
import re
import yt_dlp

# Set OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

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
    """YouTube 자막 가져오기 (yt-dlp 사용)"""
    try:
        import json
        import urllib.request

        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'ko'],
            'quiet': True,
            'no_warnings': True,
        }

        url = f"https://www.youtube.com/watch?v={video_id}"

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # 자막 우선순위: 영어 > 한국어
            for lang in ['en', 'ko']:
                # 수동 자막 먼저 시도
                if 'subtitles' in info and lang in info['subtitles']:
                    subs = info['subtitles'][lang]
                    # json3 형식 찾기
                    for sub in subs:
                        if sub.get('ext') == 'json3':
                            # URL에서 자막 다운로드
                            with urllib.request.urlopen(sub['url']) as response:
                                data = json.loads(response.read().decode('utf-8'))
                                text_parts = []
                                for event in data.get('events', []):
                                    if 'segs' in event:
                                        for seg in event['segs']:
                                            if 'utf8' in seg:
                                                text_parts.append(seg['utf8'])
                                if text_parts:
                                    return " ".join(text_parts)

                # 자동 생성 자막 시도
                if 'automatic_captions' in info and lang in info['automatic_captions']:
                    subs = info['automatic_captions'][lang]
                    for sub in subs:
                        if sub.get('ext') == 'json3':
                            with urllib.request.urlopen(sub['url']) as response:
                                data = json.loads(response.read().decode('utf-8'))
                                text_parts = []
                                for event in data.get('events', []):
                                    if 'segs' in event:
                                        for seg in event['segs']:
                                            if 'utf8' in seg:
                                                text_parts.append(seg['utf8'])
                                if text_parts:
                                    return " ".join(text_parts)

            return None
    except Exception as e:
        st.error(f"Error loading transcript: {str(e)}")
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

            # 스크립트 섹션 추가
            with st.expander("📄 View Full Transcript"):
                st.markdown("### Full English Transcript")
                st.text_area(
                    "Transcript for Shadowing Practice",
                    transcript,
                    height=400,
                    label_visibility="collapsed"
                )
                st.download_button(
                    label="⬇️ Download Transcript",
                    data=transcript,
                    file_name=f"transcript_{video_id}.txt",
                    mime="text/plain"
                )

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
                     You are an English learning assistant. The user is using this tool to study English through shadowing practice.

                     IMPORTANT: When the user asks for the script, transcript, or full text, you MUST provide it completely.
                     This is for educational purposes (language learning and shadowing practice).

                     You can:
                     - Provide the full transcript when requested
                     - Break it into sections for easier reading
                     - Summarize the content
                     - Explain difficult vocabulary or expressions
                     - Answer questions about the video content

                     The transcript is provided below. Use it to help the user study English.

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
