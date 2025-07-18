from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from PyPDF2 import PdfReader
import chromadb
from langchain_openai import OpenAIEmbeddings
import os
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from sentence_transformers import SentenceTransformer
import getpass
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from google.auth import exceptions
from groq import Groq
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

import streamlit as st

# Environment setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# Initialize components
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = init_chat_model("google_genai:gemini-2.0-flash")

vector_store = Chroma(
    collection_name="pdf_documents",  
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db", 
)

def pdf2text(location: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Process PDF and add to vector store"""
    try:
        loader = PyPDFLoader(location)
        doc = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(doc)
        print(f"Split PDF into {len(all_splits)} chunks.")

        doc_ids = vector_store.add_documents(documents=all_splits)
        print(f"Added {len(doc_ids)} documents to vector store")
        return doc_ids
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    """Retrieve relevant documents from vector store"""
    try:
        retrieved_docs = vector_store.similarity_search(
            state["question"], 
            k=5 
        )
        
        if not retrieved_docs:
            print("No relevant documents found")
            return {"context": []}
        
        print(f"Retrieved {len(retrieved_docs)} relevant documents")
        return {"context": retrieved_docs}
        
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return {"context": []}

def generate(state: State):
    """Generate answer based on retrieved context"""
    if not state["context"]:
        return {"answer": "I couldn't find relevant information to answer your question."}
    
    try:
        
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        
        prompt = f"""
        Based on the following context, please answer the question. If the answer cannot be found in the context, say so clearly.

        Context:
        {docs_content}

        Question: {state["question"]}

        Answer:
        """
        
        response = llm.invoke(prompt)
        return {"answer": response.content}
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return {"answer": "Sorry, I encountered an error while generating the answer."}

def initialize_rag_system():
    """Initialize the RAG system with graph compilation"""
    try:
        # Build the graph
        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("generate", generate)
        
        # Define the flow
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        
        graph = graph_builder.compile()
        return graph
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

def process_pdf_if_needed():
    """Process PDF if not already processed"""
    # Use session state to track if PDF has been processed
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    
    if not st.session_state.pdf_processed:
        with st.spinner("Processing PDF transcript..."):
            try:
                # Uncomment and modify as needed
                # pdf_result = pdf2text("transcipt_1.pdf")
                # if pdf_result is None:
                #     st.error("Failed to process PDF. Please check the file.")
                #     return False
                
                st.session_state.pdf_processed = True
                st.success("PDF processed successfully!")
                return True
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                return False
    return True

def main():
    """Main function to run the RAG pipeline"""
    st.title("RAG based ChatBot!")
    
    # Initialize session state for conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Initialize RAG system
    if 'graph' not in st.session_state:
        st.session_state.graph = initialize_rag_system()
    
    if st.session_state.graph is None:
        st.error("RAG system initialization failed. Please refresh the page.")
        return
    
    # Process PDF if needed
    if not process_pdf_if_needed():
        return
    
    st.write("RAG System initialized. Ask questions about the transcript!")
    
    # Chat interface
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Your question:", placeholder="Ask something about the transcript...")
        submitted = st.form_submit_button("Send")
        
        if submitted and user_input:
            try:
                # Add user message to history
                st.session_state.conversation_history.append({"role": "user", "content": user_input})
                
                # Process the question
                with st.spinner("Thinking..."):
                    state = st.session_state.graph.invoke({"question": user_input})
                    answer = state.get('answer', 'No answer generated')
                
                # Add bot response to history
                st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error processing question: {e}")
    
    # Display conversation history
    st.subheader("Conversation History")
    for i, message in enumerate(reversed(st.session_state.conversation_history)):
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**Bot:** {message['content']}")
        
        if i < len(st.session_state.conversation_history) - 1:
            st.divider()
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.conversation_history = []
        st.rerun()

if __name__ == "__main__":
    main()

    
