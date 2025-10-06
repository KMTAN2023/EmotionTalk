# EmotionTalk
Oct 2025

# MERaLiON Chatbot Live Test

This is a public-facing web application created to test a quantized version of the `LLaMA-3-MERaLiON-8B-Instruct` model. The application is built with Streamlit and LangChain, and it is deployed on a free CPU tier on Hugging Face Spaces.

## ⚠️ Performance Note

This application runs a ~5GB language model on a free, shared CPU. As a result, **response times will be slow (5-15 seconds per response)**. This is an expected trade-off for running a self-contained, powerful model on free hardware.

## Tech Stack

*   **LLM:** `mradermacher/LLaMA-3-MERaLiON-8B-Instruct-GGUF` (Q4_K_S Quant)
*   **Web Framework:** Streamlit
*   **Orchestration:** LangChain
*   **Model Runner:** CTransformers
*   **Hosting:** Hugging Face Spaces (CPU Basic)
*   **Development:** GitHub Codespaces

## How to Run

1.  **Clone the repository.**
2.  **Set up a Python virtual environment.**
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Launch the app:**
    ```bash
    streamlit run app.py
    ```
