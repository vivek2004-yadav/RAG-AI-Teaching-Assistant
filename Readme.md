# RAG AI Teaching Assistant

A Retrieval-Augmented Generation (RAG) system designed to index educational videos and playlists, allowing users to ask questions in natural language and receive grounded answers along with exact video titles and timestamps.

---

## Key Features

* **Speech-to-Text Processing:** Converts video and audio lecture files into timestamped JSON transcriptions using OpenAI Whisper.
* **Vector Embeddings:** Generates semantic embeddings for transcript text chunks using Google Gemini's Embedding API (`text-embedding-004`).
* **Fast Local Vector Caching:** Serializes pre-computed embeddings to disk using `joblib` (`embeddings_gemini.joblib`) for instant in-memory vector similarity searches without repeated cloud API calls.
* **Grounded Answer Generation:** Performs cosine similarity matching to retrieve top relevant transcript chunks and uses Google Gemini LLM (`gemini-1.5-flash`) to generate precise, hallucination-free answers with timestamp citations.
* **Interactive CLI Interface:** Allows users to query the index interactively from the terminal.

---

## Tech Stack

* **Language:** Python 3.10+
* **Speech Recognition:** OpenAI Whisper (`whisper`)
* **LLM & Embeddings:** Google Generative AI (`google-generativeai` / Gemini 1.5)
* **Vector Storage & Math:** `joblib`, `numpy`, `scikit-learn`
* **Data Format:** JSON

---

## Project Structure

```text
RAG-AI-Teaching-Assistant/
├── jsons/                    # Preprocessed JSON transcript files with timestamps
├── embeddings_gemini.joblib  # Serialized vector embeddings file
├── video_to_mp3.py           # Utility to extract audio (.mp3) from video files
├── mp3_to_json.py            # Speech-to-text pipeline using Whisper
├── preprocess_json.py        # Vector embedding generation & joblib serialization
├── process_incoming.py       # Main interactive RAG search interface
├── prompt.txt                # System prompt instructions template
└── Readme.md                 # Project documentation
```

---

## How to Run the Assistant

1. Ensure Python 3.10 or higher is installed.
2. Install required Python packages:
   ```bash
   pip install google-generativeai joblib numpy scikit-learn openai-whisper
   ```
3. Set your Google Gemini API Key as an environment variable:
   ```powershell
   $env:GEMINI_API_KEY="your_actual_api_key"
   ```
4. Run the interactive search interface:
   ```bash
   python process_incoming.py
   ```
5. Type your question when prompted (e.g., *"What is CSS Box Model?"*) to receive the answer and timestamp references.

---

## How to Build the Index for New Videos

To index your own videos or playlists:

1. **Convert Video to MP3:** Place video files in the root folder and run:
   ```bash
   python video_to_mp3.py
   ```
2. **Transcribe Audio to JSON:** Run Whisper to generate timestamped text transcripts in the `jsons/` folder:
   ```bash
   python mp3_to_json.py
   ```
3. **Generate & Cache Vector Embeddings:** Convert transcript chunks into Gemini embeddings and save them to `embeddings_gemini.joblib`:
   ```bash
   python preprocess_json.py
   ```
