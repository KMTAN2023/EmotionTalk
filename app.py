# app.py (Final Corrected Version)
import streamlit as st
import os
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import hf_hub_download

# --- 1. Model Loading ---
# This is cached to load the model only once.
@st.cache_resource
def load_llm_model():
    """
    Downloads the model from Hugging Face and loads it using LlamaCpp.
    """
    # Step 1: Download the model file from Hugging Face Hub
    model_name = "mradermacher/LLaMA-3-MERaLiON-8B-Instruct-GGUF"
    # Using the smaller Q3_K_L model as requested
    model_file = "LLaMA-3-MERaLiON-8B-Instruct.Q3_K_L.gguf"
    model_path = hf_hub_download(repo_id=model_name, filename=model_file)

    # Step 2: Load the model using the downloaded path
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0, # Run on CPU
        n_threads=os.cpu_count() or 1,
        temperature=0.7,
    )
    return llm

# Load the model when the app starts
llm = load_llm_model()

# --- 2. Streamlit UI and Chat Logic ---
st.title("🤖 MERaLiON Chatbot Test (Q3_K_L Model)")
st.warning("This demo runs a ~4.4GB LLM on a free CPU. Responses will be slow. Please be patient!", icon="⏳")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am running the smaller Q3_K_L model. How can I help you today?"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("MERaLiON (Q3) is thinking..."):
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful Singaporean assistant. Answer the user's question concisely."),
                ("user", "{question}")
            ])
            output_parser = StrOutputParser()
            chain = prompt | llm | output_parser
            
            response = chain.invoke({"question": user_prompt})
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
