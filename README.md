# 📄 AI Document Q&A Agent

An intelligent web application that allows users to upload PDF documents and ask questions about their content. Built with Retrieval-Augmented Generation (RAG) technology, it extracts text, generates embeddings, and uses an LLM to provide precise, context-aware answers.

## 🚀 Features
- **PDF Processing:** Extracts text from single or multiple uploaded PDFs.
- **Smart Chunking:** Uses LangChain to split massive documents into readable segments.
- **Vector Search:** Converts text into embeddings using Google Generative AI and stores them locally via FAISS.
- **Conversational AI:** Answers user queries strictly based on the uploaded document context, preventing hallucinations.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Framework:** LangChain
- **AI Models:** Google Gemini 1.5 Flash & Google Embeddings
- **Vector Database:** FAISS

## 💻 How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/Khaja17-ai/AI-Project.git](https://github.com/Khaja17-ai/AI-Project.git)
   cd AI-Project