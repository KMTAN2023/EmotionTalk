# app.py
import streamlit as st
import os
from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. Model Loading ---
# This is cached to load the 4.8GB model only once.
@st.cache_resource
def load_llm_model():
    """Loads the MERaLiON GGUF model from Hugging Face, configured for CPU."""
    llm = CTransformers(
        # The repository you found
        model="mradermacher/LLaMA-3-MERaLiON-8B-Instruct-GGUF",
        # The specific quantized file we chose
        model_file="LLaMA-3-MERaLiON-8B-Instruct.Q4_K_S.gguf",
        model_type="llama",
        # Configuration for CPU inference
        config={
            'context_length': 2048, 
            'temperature': 0.7,
            'threads': os.cpu_count() or 1 # Use all available CPU threads
        }
    )
    return llm

# Load the model when the app starts
llm = load_llm_model()

# --- 2. Streamlit UI and Chat Logic ---
st.title("🤖 MERaLiON Chatbot Test")
st.warning("This demo runs a ~5GB LLM on a free CPU. Responses will be slow (5-15 seconds). Please be patient!", icon="⏳")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("MERaLiON is thinking..."):
            # Create a simple LangChain chain
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful Singaporean assistant. Answer the user's question concisely."),
                ("user", "{question}")
            ])
            output_parser = StrOutputParser()
            chain = prompt | llm | output_parser
            
            # Invoke the chain to get a response
            response = chain.invoke({"question": user_prompt})
            
            st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
