# 🎓 StudyMate AI — GenAI Smart Study Assistant

> Transform your study notes into summaries, quizzes, concept explanations, and an AI-powered Q&A chatbot — all running locally with no cloud dependency.

---

## 📌 What It Does

Upload any PDF or TXT study material and instantly get:

| Feature | Description |
|---------|-------------|
| 📋 **Smart Summary** | Claude generates structured summaries with sections, bullet points, and key takeaways |
| 📝 **Quiz Generator** | Exam-level MCQ questions with scenarios, explanations, and visual aids |
| 🔑 **Key Concepts** | BERT extracts named entities · Claude explains any concept · ANN classifies sentences |
| 💬 **Chat with Notes** | Multi-tool RAG agent answers questions grounded only in your uploaded notes |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| NLP Preprocessing | NLTK · spaCy · Regex |
| Text Classification | PyTorch · ANN · CNN · LSTM |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Named Entity Recognition | BERT (`dbmdz/bert-large-cased-finetuned-conll03-english`) |
| Vector Search | FAISS |
| Generative AI | Anthropic Claude API (`claude-haiku-4-5`) |
| Summarization | Claude API (structured prompt engineering) |
| RAG | FAISS retrieval + Claude generation |

---

## 📁 Project Structure

```
StudyMateAI/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # All dependencies
│
├── preprocessing/                  # NLP pipeline
│   ├── cleaner.py                  # Text cleaning and noise removal
│   ├── tokenizer.py                # Tokenization and stopword removal
│   ├── lemmatizer.py               # Lemmatization with POS tagging
│   └── pipeline.py                 # Full pipeline orchestration
│
├── embeddings/                     # Vector representations
│   ├── one_hot.py                  # One-hot and Bag-of-Words encoding
│   └── sentence_embeddings.py      # SentenceTransformer embeddings
│
├── models/                         # Text classification
│   ├── training_data.py            # 360+ labeled training sentences
│   ├── data_prep.py                # Dataset preparation and splitting
│   ├── ann_model.py                # Artificial Neural Network
│   ├── cnn_model.py                # Convolutional Neural Network
│   ├── lstm_model.py               # Long Short-Term Memory Network
│   ├── evaluator.py                # Performance metrics (F1, Accuracy)
│   ├── train_all.py                # Master training script
│   └── saved/                      # Trained model weights (.pt files)
│
├── ner/
│   └── ner_extractor.py            # BERT-based Named Entity Recognition
│
├── generative/                     # AI content generation
│   ├── summarizer.py               # Structured summarization via Claude
│   ├── quiz_generator.py           # MCQ generation via Claude
│   └── explainer.py                # Concept explanation via Claude
│
├── rag/                            # Retrieval-Augmented Generation
│   ├── indexer.py                  # Chunk text → embed → FAISS index
│   ├── retriever.py                # Semantic search over FAISS
│   ├── qa_chain.py                 # Direct Q&A chain (fallback)
│   └── agent.py                    # Multi-tool agentic RAG loop
│
├── utils/
│   ├── pdf_reader.py               # PDF and TXT text extraction
│   └── file_handler.py             # Save/load processed data
│
└── data/
    ├── raw/                        # Uploaded files (auto-populated)
    └── processed/                  # FAISS index and chunks (auto-populated)
```

---

## ⚙️ Setup & Installation

### Requirements
- Python 3.11 (not 3.12+)
- macOS / Linux / Windows

### Steps

```bash
# 1. Clone and enter the project
cd StudyMateAI

# 2. Create virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 5. Train the classifiers (one-time, ~5 minutes)
python -m models.train_all

# 6. Launch the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🧠 How RAG Works

RAG (Retrieval-Augmented Generation) is the engine behind the Chat tab.

```
Your Notes
    │
    ▼
Split into chunks → Embed with SentenceTransformer → Store in FAISS
                                                            │
                                            ┌───────────────┘
                                            │  At query time:
User Question ──► Embed ──► FAISS Search ──► Top 5 relevant chunks
                                            │
                                            ▼
                              Claude reads ONLY those chunks
                                            │
                                            ▼
                         Structured answer grounded in your notes ✅
```

**Why it matters:** Claude cannot hallucinate — it only answers from what your notes actually say. If the topic isn't in your notes, it says so.

---

## 🤖 Where Models Are Used

| Model | Location | Purpose |
|-------|----------|---------|
| ANN / CNN / LSTM | `models/` → Tab 3 | Classify sentences as Definition / Concept / Example / Important Point |
| SentenceTransformer | `embeddings/` → everywhere | Convert text to 384-dim vectors for FAISS and classification |
| BERT NER | `ner/` → Tab 3 | Extract People, Organizations, Locations from notes |
| Claude API | `generative/` + `rag/` | Summarization, Quiz generation, Explanation, Chat answers |
| FAISS | `rag/` → Tab 4 | Store and search vector embeddings for RAG |

---

## 👥 Team Contributions

| Member | Roll | Contribution |
|--------|------|-------------|
| **Ashi Srivastava** | — | **Data collection and NLP preprocessing pipeline** — tokenization, stopword removal, lemmatization, and text cleaning modules |
| **Parth Nawal** | — | **Embeddings and Named Entity Recognition** — SentenceTransformer integration, one-hot encoding, BERT NER implementation |
| **Simran Karan Bora** | — | **Deep learning models** — design, training, and evaluation of ANN, CNN, and LSTM classifiers with F1-score comparison |
| **Vidushi Bhadauria** | — | **Application integration** — main app.py, Streamlit UI, module wiring, and complete project compilation |
| **Aryama Sharma** | — | **Generative AI and RAG** — summarizer, quiz generator, concept explainer, FAISS indexing, and multi-tool agent |

---

## 📊 Model Performance (after training)

| Model | Accuracy | F1-Score |
|-------|----------|---------|
| ANN   | ~88%     | ~0.87   |
| CNN   | ~90%     | ~0.89   |
| LSTM  | ~88%     | ~0.87   |

*Trained on 500+ samples across 12 academic subjects.*

---

*Built with Python · NLP · HuggingFace · FAISS · Anthropic Claude*
