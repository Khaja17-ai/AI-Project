import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AI Document Agent", page_icon="📄", layout="wide")
st.title("📄 AI Document Q&A Agent")
st.write("Upload a PDF, let the AI read it, and ask any question about its contents.")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key == "your_api_key_here":
    st.error("⚠️ Please add your Gemini API Key to the .env file to continue.")
    st.stop()

# --- SIDEBAR: Uploading & Processing ---
with st.sidebar:
    st.header("1. Upload Documents")
    pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type=["pdf"])
    process_button = st.button("Process Documents")

# --- CORE LOGIC: Reading and Embedding ---
if process_button and pdf_docs:
    with st.spinner("Reading, chunking, and learning the documents..."):
        raw_text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_text(raw_text)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
        
        # Storing vectors safely in Streamlit's session memory
        st.session_state.vector_store = InMemoryVectorStore.from_texts(text_chunks, embedding=embeddings)
        
        st.success("✅ Documents learned! You can now ask questions.")

# --- UI: Chat / Question Area ---
st.markdown("---")
st.header("2. Ask Questions")
user_question = st.text_input("What do you want to know from the documents?")

if user_question:
    if "vector_store" not in st.session_state:
        st.warning("⚠️ Please upload and process a document first.")
    else:
        with st.spinner("Thinking..."):
           # 1. Retrieve more text chunks (k=8) so it has enough context to actually summarize
            docs = st.session_state.vector_store.similarity_search(user_question, k=8)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # 2. Give the AI a specific "Tutor/Study Buddy" Persona
            prompt_template = """
            You are a friendly, highly intelligent study buddy and AI tutor. 
            Use the following context from the uploaded document to answer the user's question.
            
            - If the user asks for a summary, explain it simply and conversationally.
            - If the user uploads an assignment or syllabus and asks for advice, analyze the text to identify the most important concepts. 
            - Suggest a step-by-step study plan (e.g., "Study this first, then look at this next").
            - Always maintain a helpful, encouraging, and conversational tone, like a smart friend helping them prepare for an exam.
            
            Context:
            {context}
            
            Question: 
            {question}
            
            Answer:
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
            # 3. Initialize Gemini
            model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
            
           # 4. Modern LCEL approach: Pipe through the string parser
            chain = prompt | model | StrOutputParser()
            
            # 5. Display the AI Response using Streaming
            st.write("### AI Response:")
            
            # st.write_stream automatically catches the words as they generate and types them on screen
            with st.spinner("Thinking..."):
                response_stream = chain.stream({"context": context_text, "question": user_question})
                st.write_stream(response_stream)