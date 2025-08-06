import os
import cv2
import torch
import time
import subprocess
import numpy as np
import tempfile
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.llms import HuggingFacePipeline

# ✅ Path to ffprobe.exe
ffprobe_path = r"C:\Users\adesh kachare\PyCharmMiscProject\Visual_Understanding_Chatbot\app\ffmpeg-7.1.1-essentials_build\bin\ffprobe.exe"

# ✅ Fix: Accept video_path as argument properly
def get_video_duration(video_path):
    try:
        result = subprocess.run([
            ffprobe_path,
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        st.error(f"Failed to extract video duration: {e}")
        return 0

@st.cache_resource
def load_yolo_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5n', trust_repo=True)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, return_full_text=False)
    return HuggingFacePipeline(pipeline=pipe)

def detect_objects_in_frame(frame, model):
    results = model(frame)
    return results.pandas().xyxy[0]

def summarize_events(violations):
    if not violations:
        return "No significant events or violations detected."
    summary = "Summary of detected events and violations:\n"
    for i, v in enumerate(violations):
        summary += f"{i+1}. {v}\n"
    return summary

def extract_frames(video_path, sample_rate=1):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames, timestamps = [], []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % (fps * sample_rate) == 0:
            frames.append(frame)
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        count += 1
    cap.release()
    return frames, timestamps

def detect_events(frames, timestamps):
    model = load_yolo_model()
    events, snapshots = [], []
    for frame, ts in zip(frames, timestamps):
        results = detect_objects_in_frame(frame, model)
        people = results[results['name'] == 'person']
        if len(people) > 3:
            events.append(f"Crowd detected at {ts:.2f}s with {len(people)} people.")
            snapshots.append(frame)
    return events, snapshots

def create_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(text_chunks, embeddings)

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents([Document(page_content=text)])

# ---------------------------- Streamlit UI ---------------------------- #
st.set_page_config(page_title="Visual Understanding Assistant", layout="wide")
st.title("\U0001F4F9 Visual Understanding Chat Assistant")

uploaded_video = st.file_uploader("Upload a video for analysis (max 2 minutes)", type=["mp4", "avi", "mov"])

if uploaded_video:
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_video.read())
    video_path = temp_video.name

    duration = get_video_duration(video_path)
    if duration > 130:
        st.warning("❌ Video is longer than 2 minutes. Please upload a shorter video.")
    else:
        st.video(video_path)
        st.info("⏳ Analyzing video... This may take up to 1 minute.")

        frames, timestamps = extract_frames(video_path)
        events, snapshots = detect_events(frames, timestamps)
        summary = summarize_events(events)

        st.subheader("🧠 Event Summary")
        st.success(summary)

        st.subheader("🖼️ Snapshots of Detected Events")
        for i, snap in enumerate(snapshots):
            st.image(snap, caption=f"Snapshot {i+1}", channels="BGR")

        # Chat Assistant Section
        st.subheader("🤖 Ask Questions")
        text_chunks = split_text(summary)
        vectordb = create_vector_store(text_chunks)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=load_llm(),
            retriever=vectordb.as_retriever(),
            memory=memory
        )

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Ask anything about the video")
        if user_input:
            response = qa_chain.run(user_input)
            st.session_state.chat_history.append((user_input, response))

        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")
