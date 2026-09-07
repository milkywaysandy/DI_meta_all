# Drug RAG Assistant

A secure, context-aware Retrieval-Augmented Generation (RAG) prototype built with **Streamlit**, **LangChain**, and **Google Gemini**, designed to query medical package inserts and pharmaceutical data in Traditional Chinese.

---

## **Key Features**

* **Gemini-Powered Generation:** Leverages state-of-the-art Google models (`gemini-2.5-flash` and `gemini-embedding-001`) for precise medical data synthesis.
* **Smart Retrieval:** Uses FAISS vector search with Maximal Marginal Relevance (MMR) routing to pull relevant drug documentation.
* **Multi-Variant Handling:** Automatically synthesizes information across multiple drug formulations or families while citing source files.
* **Robust Error Handling:** Features exponential backoff retry logic and clean debug logging.

---

## **Prerequisites & Setup**

1. Python 3.10 or higher.
2. A valid Google Generative AI API Key.

### **1. Installation**

Clone the repository and install the required dependencies:

```bash
pip install streamlit langchain-community langchain-google-genai google-generativeai faiss-cpu

```

### **2. Configuration**

This app safely manages credentials using Streamlit secrets or environment variables. Create a `.streamlit/secrets.toml` file in your project root:

```toml
GOOGLE_API_KEY = "your_actual_api_key_here"

```

Alternatively, set it via your environment:

```bash
export GOOGLE_API_KEY="your_actual_api_key_here"

```

### **3. Vector Database Preparation**

Ensure your local FAISS vector store folder (`faiss_index_medical`) is placed in the root directory of the application before launching.

---

## **Running the Application**

Start the Streamlit server locally by running:

```bash
streamlit run app.py

```

---

## **Usage Guide**

1. Open the local URL provided in your terminal (typically `http://localhost:8501`).
2. Input your medical query into the text box (e.g., *what is the possible fetal toxicity when taking Ergonovine Maleate?*).
3. Click **Generate Answer** to view the synthesized response and expand the **Source Documents** section to verify the exact document snippets utilized.
