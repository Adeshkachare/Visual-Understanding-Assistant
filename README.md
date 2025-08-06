
# 📹 Visual Understanding Chat Assistant

A real-time visual understanding system that analyzes short videos (≤2 mins), detects events and violations using object detection (YOLOv5), summarizes key incidents, and enables conversational interaction with the video using Mistral-7B-based LLM.

![Demo](https://youtu.be/Y1QyB7cpzc0)

---

## 🚀 Features

- 🎥 **Video Upload** – Supports `.mp4`, `.avi`, `.mov` formats up to 2 minutes.
- 🧠 **Event Detection** – Detects crowding or custom events using YOLOv5n.
- 📝 **Summarization** – Generates clear, structured summaries of detected events.
- 💬 **Conversational AI** – Ask questions about the video and get relevant responses.
- 🖼️ **Snapshots** – Shows event-based video snapshots with timestamps.
- 🔎 **Semantic Search + RAG** – Uses HuggingFace embeddings and FAISS for contextual retrieval.
- ⚡ **Fast Inference** – Uses the `mistralai/Mistral-7B-Instruct-v0.2` model with GPU acceleration.
- 🎛️ **Streamlit UI** – Interactive, fast, and intuitive frontend.

---

## 🛠️ Tech Stack

| Component              | Technology                                  |
|------------------------|---------------------------------------------|
| UI/Frontend            | Streamlit                                   |
| Object Detection       | YOLOv5n via Torch Hub                        |
| Embedding Model        | `all-MiniLM-L6-v2` (SentenceTransformer)     |
| Language Model (LLM)   | `mistralai/Mistral-7B-Instruct-v0.2`         |
| Text Chunking          | LangChain `RecursiveCharacterTextSplitter`  |
| Vector DB              | FAISS (in-memory)                           |
| Chat Memory            | LangChain `ConversationBufferMemory`        |
| Video Processing       | OpenCV + ffmpeg (via subprocess)            |

---

## ⚙️ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/your_username/visual-understanding-chat-assistant.git
cd visual-understanding-chat-assistant
```

### 2. Create Environment & Install Dependencies
```bash
pip install -r requirements.txt
```

#### Example `requirements.txt`:
```txt
streamlit
torch
opencv-python
transformers
sentence-transformers
langchain
faiss-cpu
```

> ⚠️ You must have `ffmpeg` and `ffprobe` installed and available on your system. Update the `ffprobe_path` in `main_app.py` accordingly.

### 3. Run the App
```bash
streamlit run main_app.py
```

---

## 📂 Project Structure

```
├── main_app.py                # Main Streamlit application
├── README.md                  # Project overview
├── requirements.txt           # Required Python packages
├── /ffmpeg-bin/               # ffprobe & ffmpeg executables
├── /snapshots/ (optional)     # Saved frames of detected events
```

---

## 🧪 How It Works

1. **Upload** a video file (max 2 mins).
2. **Extract Frames** at 1 fps for efficient processing.
3. **Detect Objects** in each frame using YOLOv5n.
4. **Flag Events** such as crowding (>3 people).
5. **Summarize Events** as text.
6. **Embed + Store** summaries in a vector DB.
7. **Chat** with a retrieval-augmented Mistral-7B chatbot about the video.

---

## 🤖 Example Questions to Ask the Assistant

- *"What happened at the 1-minute mark?"*
- *"Were there any large crowds?"*
- *"Give a brief overview of the video."*
- *"How many people were detected overall?"*

---

## 🧠 Model Info

- 🔹 Mistral-7B: Efficient instruction-tuned model running with GPU acceleration.
- 🔹 Embeddings: `all-MiniLM-L6-v2` (lightweight and fast).
- 🔹 Detection: `yolov5n` for fast frame-level inference.

---

## 🧩 Limitations

- Currently supports basic event detection (e.g., crowding).
- LLM is locally hosted and requires sufficient VRAM.
- Video length is capped at 2 minutes for latency and performance.
- Doesn’t support multi-class rule violations yet.

---

## 📌 To-Do / Improvements

- [ ] Add custom violation logic (e.g., helmet detection, red light).
- [ ] Support for longer video processing with async inference.
- [ ] Integrate with real-time streams (e.g., RTSP, webcam).
- [ ] Optimize memory & improve snapshot tagging.
- [ ] Deployment using Docker or HuggingFace Spaces.

---

## 💡 Inspiration

Built as part of **Mantra Hackathon 2025** to explore real-time video understanding and multi-modal AI agents.

---

## 📬 Contact

- **Developer**: Adesh Kachare  
- **Email**: your.email@example.com  
- **LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)  
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
