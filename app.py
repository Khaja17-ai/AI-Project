import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables (API Key) from the .env file
load_dotenv()

# Configure the Streamlit UI
st.set_page_config(page_title="AI Document Agent", page_icon="📄", layout="wide")
st.title("📄 AI Document Q&A Agent")
st.write("Upload a PDF, let the AI read it, and ask any question about its contents.")

# Ensure API key is present
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
        # Extract text from all PDFs
        raw_text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    raw_text += text
        
        # Split the massive text into readable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_text(raw_text)
        
        # Convert text into searchable AI vectors using Google Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # Save the vectors locally so we can search them
        vector_store.save_local("faiss_index")
        st.success("✅ Documents learned! You can now ask questions.")

# --- UI: Chat / Question Area ---
st.markdown("---")
st.header("2. Ask Questions")
user_question = st.text_input("What do you want to know from the documents?")

if user_question:
    if not os.path.exists("faiss_index"):
        st.warning("⚠️ Please upload and process a document first.")
    else:
        with st.spinner("Thinking..."):
            # Load the AI embeddings and the local database
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            # Find the most relevant chunks of text
            docs = new_db.similarity_search(user_question)
            
            # Formulate the prompt rules
            prompt_template = """
            Answer the question as detailed as possible using ONLY the provided context. 
            If the answer is not in the context, do not guess. Just say, "The answer is not available in the uploaded document."
            
            Context:
            {context}
            
            Question: 
            {question}
            
            Answer:
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
            # Initialize Gemini AI
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            
            # Get the final answer
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            
            # Display the output
            st.write("### AI Response:")
            st.info(response["output_text"])