

import streamlit as st
import os
import google.generativeai as genai
# from google import genai # Removed conflicting import
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import sys # Import sys for printing to stderr

# --- Configuration & Initialization ---

# GOOGLE_API_KEY is already defined in a previous cell (4qtvFVYF3iMr)
# and passed into the function. Removing redundant definition here.
GOOGLE_API_KEY="AIzaSyBgY5Fr0FTwJ3ys_VFXvPeyvKkVssfIGY0"
@st.cache_resource
def initialize_rag_components(api_key, _preloaded_vectorstore=None):
    print("DEBUG: Entering initialize_rag_components", file=sys.stderr)
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004")
        print("DEBUG: Embeddings model initialized", file=sys.stderr)

        vectorstore_to_use = None
        if _preloaded_vectorstore is not None:
            vectorstore_to_use = _preloaded_vectorstore
            print("DEBUG: Using pre-loaded FAISS vector store.", file=sys.stderr)
            vectorstore_path_to_load = "/faiss_index_medical/"
        else:
            print("DEBUG: Attempting to load FAISS vector store locally/from GDrive.", file=sys.stderr)
            # Define potential paths
            streamlit_local_vectorstore_path = "faiss_index_medical" # For Streamlit deployment (relative to app.py)
            gdrive_vectorstore_path = "/content/gdrive/MyDrive/NTU_work/insert_rag/faiss_index_medical" # For Colab/GDrive

            vectorstore_path_to_load = "/faiss_index_medical/"

            # Prioritize local path for Streamlit flexibility
            if os.path.exists(streamlit_local_vectorstore_path):
                vectorstore_path_to_load = streamlit_local_vectorstore_path
                print(f"DEBUG: Found vector store at local path: {vectorstore_path_to_load}", file=sys.stderr)
            elif os.path.exists(gdrive_vectorstore_path):
                vectorstore_path_to_load = gdrive_vectorstore_path
                print(f"DEBUG: Found vector store at GDrive path: {vectorstore_path_to_load}", file=sys.stderr)

            if vectorstore_path_to_load is None:
                st.error(f"Error: The vectorstore folder was not found in either '{streamlit_local_vectorstore_path}' or '{gdrive_vectorstore_path}'. Please ensure it's in one of these locations.")
                # Temporarily raise a Python exception for Colab debugging
                raise FileNotFoundError(f"Vectorstore folder not found at {streamlit_local_vectorstore_path} or {gdrive_vectorstore_path}")

            vectorstore_to_use = FAISS.load_local(
                folder_path=vectorstore_path_to_load,
                embeddings=embeddings_model,
                allow_dangerous_deserialization=True
            )
            print(f"DEBUG: Loaded FAISS vector store from: {vectorstore_path_to_load}", file=sys.stderr)

        retriever = vectorstore_to_use.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        print("DEBUG: Retriever created", file=sys.stderr)

        # Explicitly check for genai.configure before calling it
        if not hasattr(genai, 'configure'):
            raise AttributeError("module 'google.generativeai' has no attribute 'configure'. Check imports and environment.")

        genai.configure(api_key=api_key)
        print("DEBUG: genai configured", file=sys.stderr)
        llm_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={"temperature": 0}
        )
        print("DEBUG: LLM model created", file=sys.stderr)
        print("DEBUG: Exiting initialize_rag_components successfully", file=sys.stderr)
        return retriever, llm_model
    except Exception as e:
        print(f"DEBUG: Exception during RAG initialization: {e}", file=sys.stderr) # This will print to the kernel's stderr
        st.error(f"Error during RAG initialization: {e}")
        # Temporarily re-raise the exception for Colab debugging
        raise e

# --- RAG Logic Function ---

def run_rag_pipeline(query: str, retriever, llm_model):
    """Executes the RAG sequence."""

    # 1. Retrieval
    most_similar_documents = retriever.invoke(query)

    # 2. Context Formatting
    context_parts = []
    for doc in most_similar_documents:
        source = doc.metadata.get('source_file', 'N/A')
        content = doc.page_content
        context_parts.append(f"Source: {source}\nContent: {content}")
    context_text = "\n\n".join(context_parts)

    # 3. Define the Template
    base_prompt = """You are a medical information assistant analyzing drug package inserts... [rest of your prompt]...\n    Context:\n    {context}\n\n    ANSWER REQUIREMENTS:\n    - Be precise and cite specific drug names for each piece of information\n    - Use Traditional Chinese in your response\n    - If comparing multiple drugs or variants, organize your answer clearly by drug name\n    - If information is missing for a specific drug, state this clearly\n    - For safety-critical information (contraindications, interactions, dosage), be extra thorough\n    - Demonstrate your reasoning by citing the chunks (say their header names instead of chunks if you can) that made your conclusion and why that forms your conclusion\n\n    Question: {question}\n    Helpful Answer:"""

    # 4. Inject Context and Question
    final_prompt = base_prompt.format(context=context_text, question=query)

    # 5. Generation
    response = llm_model.generate_content(final_prompt)
    return response.text, most_similar_documents

# --- Streamlit Application Layout ---

st.title("💊 Prototype Drug RAG Assistant")
st.markdown("Enter a medical query to retrieve relevant drug information and generate an answer.")

# Initialize RAG components.
# Pass the globally loaded 'vectorstore' if it exists.
# Otherwise, initialize_rag_components will attempt to load locally.
# 'vectorstore' is defined in cell 6EdB-EuF9xGt
if 'vectorstore' in globals(): # Check if global vectorstore is defined
    retriever_instance, llm_model_instance = initialize_rag_components(GOOGLE_API_KEY, _preloaded_vectorstore=vectorstore)
else:
    retriever_instance, llm_model_instance = initialize_rag_components(GOOGLE_API_KEY)


# User Input
query = st.text_input(
    "Your Medical Question:",
    placeholder="e.g., what is the possible fetal toxicity when taking Ergonovine Maleate?"
)

if st.button("Generate Answer"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching and generating answer..."):
            answer, documents = run_rag_pipeline(query, retriever_instance, llm_model_instance)

            st.subheader("Generated Answer")
            st.info(answer)

            with st.expander("Source Documents (Context Used)"):
                for i, doc in enumerate(documents):
                    st.markdown(f"**Document {i+1}** (Source: `{doc.metadata.get('source_file', 'N/A')}`)")
                    st.code(doc.page_content[:500] + "...", language='text')











