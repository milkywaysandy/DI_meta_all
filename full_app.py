import streamlit as st
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import sys
import time

# --- Configuration & Initialization ---

# ⚠️ SECURITY WARNING: Never hardcode API keys in source code!
# Use Streamlit secrets instead:
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback for local testing only - Remove before production
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("❌ GOOGLE_API_KEY not found.  Please set it in Streamlit secrets or environment variables.")
        st.stop()

@st.cache_resource
def initialize_rag_components(api_key, _preloaded_vectorstore=None):
    """Initialize RAG components with improved error handling."""
    print("DEBUG: Entering initialize_rag_components", file=sys.stderr)
    try:
        # Validate API key format
        if not api_key or len(api_key) < 10:
            raise ValueError("Invalid API key format")
        
        print("DEBUG: Initializing embeddings model.. .", file=sys.stderr)
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=api_key  # Explicitly pass API key
        )
        print("DEBUG: Embeddings model initialized", file=sys.stderr)

        vectorstore_to_use = None
        if _preloaded_vectorstore is not None:
            vectorstore_to_use = _preloaded_vectorstore
            print("DEBUG: Using pre-loaded FAISS vector store.", file=sys.stderr)
        else:
            print("DEBUG: Attempting to load FAISS vector store locally/from GDrive.", file=sys.stderr)
            # Define potential paths
            streamlit_local_vectorstore_path = "faiss_index_medical"
            gdrive_vectorstore_path = "/content/gdrive/MyDrive/NTU_work/insert_rag/faiss_index_medical"

            vectorstore_path_to_load = None

            # Prioritize local path for Streamlit flexibility
            if os.path.exists(streamlit_local_vectorstore_path):
                vectorstore_path_to_load = streamlit_local_vectorstore_path
                print(f"DEBUG: Found vector store at local path: {vectorstore_path_to_load}", file=sys.stderr)
            elif os.path.exists(gdrive_vectorstore_path):
                vectorstore_path_to_load = gdrive_vectorstore_path
                print(f"DEBUG: Found vector store at GDrive path: {vectorstore_path_to_load}", file=sys.stderr)

            if vectorstore_path_to_load is None:
                error_msg = f"Vectorstore folder not found at '{streamlit_local_vectorstore_path}' or '{gdrive_vectorstore_path}'"
                st.error(f"❌ Error: {error_msg}")
                print(f"DEBUG: {error_msg}", file=sys. stderr)
                raise FileNotFoundError(error_msg)

            print(f"DEBUG: Loading FAISS from {vectorstore_path_to_load}...", file=sys.stderr)
            vectorstore_to_use = FAISS.load_local(
                folder_path=vectorstore_path_to_load,
                embeddings=embeddings_model,
                allow_dangerous_deserialization=True
            )
            print(f"DEBUG: Loaded FAISS vector store from: {vectorstore_path_to_load}", file=sys.stderr)

        retriever = vectorstore_to_use.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        print("DEBUG: Retriever created", file=sys.stderr)

        # Configure genai
        if not hasattr(genai, 'configure'):
            raise AttributeError("module 'google.generativeai' has no attribute 'configure'")

        genai.configure(api_key=api_key)
        print("DEBUG: genai configured", file=sys.stderr)
        
        llm_model = genai.GenerativeModel(
            model_name="gemini-2. 5-flash",
            generation_config={"temperature": 0}
        )
        print("DEBUG: LLM model created", file=sys.stderr)
        print("DEBUG: Exiting initialize_rag_components successfully", file=sys.stderr)
        return retriever, llm_model, embeddings_model
        
    except Exception as e:
        error_msg = f"RAG Initialization Error: {str(e)}"
        print(f"DEBUG: Exception during RAG initialization: {error_msg}", file=sys.stderr)
        st.error(f"❌ {error_msg}")
        raise

# --- RAG Logic Function ---

def run_rag_pipeline(query: str, retriever, llm_model, embeddings_model=None):
    """Executes the RAG sequence with retry logic."""
    
    max_retries = 2
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            print(f"DEBUG: RAG pipeline attempt {attempt + 1}/{max_retries} for query: {query[:50]}...", file=sys. stderr)
            
            # 1. Retrieval with error handling
            print("DEBUG: Starting similarity search.. .", file=sys.stderr)
            most_similar_documents = retriever.invoke(query)
            print(f"DEBUG: Retrieved {len(most_similar_documents)} documents", file=sys.stderr)
            
            if not most_similar_documents:
                st.warning("⚠️ No relevant documents found for your query.")
                return "No relevant information found in the knowledge base.", []

            # 2. Context Formatting
            context_parts = []
            for i, doc in enumerate(most_similar_documents):
                source = doc.metadata.get('source_file', 'N/A')
                content = doc.page_content
                context_parts.append(f"Source: {source}\nContent: {content}")
                print(f"DEBUG: Document {i+1} source: {source}", file=sys. stderr)
            
            context_text = "\n\n".join(context_parts)

            # 3. Define the Template
            base_prompt = """You are a medical information assistant analyzing drug package inserts. 
    
    Context:
    {context}
    
    ANSWER REQUIREMENTS:
    - Be precise and cite sources from the context
    - If information is not in the context, say so explicitly
    - Structure your answer clearly with key points
    
    User Question:
    {question}
    
    Answer:"""

            # 4. Inject Context and Question
            final_prompt = base_prompt.format(context=context_text, question=query)

            # 5. Generation
            print("DEBUG: Generating response from LLM...", file=sys.stderr)
            response = llm_model.generate_content(final_prompt)
            print("DEBUG: Response generated successfully", file=sys.stderr)
            return response.text, most_similar_documents
            
        except Exception as e:
            error_msg = str(e)
            print(f"DEBUG: Attempt {attempt + 1} failed: {error_msg}", file=sys.stderr)
            
            if attempt < max_retries - 1:
                st.warning(f"⚠️ Retrying ({attempt + 1}/{max_retries})...")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                print(f"DEBUG: All retries exhausted. Final error: {error_msg}", file=sys.stderr)
                error_details = f"RAG Pipeline Error: {error_msg}"
                
                if "embedding" in error_msg.lower():
                    error_details += "\n\n💡 Troubleshooting: Check your API key and rate limits."
                elif "authentication" in error_msg.lower():
                    error_details += "\n\n💡 Troubleshooting: Verify your Google API credentials."
                
                st.error(f"❌ {error_details}")
                raise

# --- Streamlit Application Layout ---

st.set_page_config(page_title="Drug RAG Assistant", layout="wide")
st.title("💊 Prototype Drug RAG Assistant")
st.markdown("Enter a medical query to retrieve relevant drug information and generate an answer.")

# Initialize RAG components with error handling
try:
    print("DEBUG: Starting RAG initialization...", file=sys.stderr)
    
    if 'vectorstore' in globals():
        retriever_instance, llm_model_instance, embeddings_instance = initialize_rag_components(
            GOOGLE_API_KEY, 
            _preloaded_vectorstore=globals().get('vectorstore')
        )
    else:
        retriever_instance, llm_model_instance, embeddings_instance = initialize_rag_components(GOOGLE_API_KEY)
    
    print("DEBUG: RAG initialization complete", file=sys.stderr)
    
except Exception as e:
    st.error(f"❌ Failed to initialize RAG system: {str(e)}")
    print(f"DEBUG: Initialization failed: {str(e)}", file=sys.stderr)
    st.stop()

# User Input
query = st.text_input(
    "Your Medical Question:",
    placeholder="e.g., what is the possible fetal toxicity when taking Ergonovine Maleate?"
)

if st.button("Generate Answer"):
    if not query or query.strip() == "":
        st.warning("❌ Please enter a question.")
    else:
        with st.spinner("🔍 Searching and generating answer..."):
            try:
                answer, documents = run_rag_pipeline(
                    query, 
                    retriever_instance, 
                    llm_model_instance, 
                    embeddings_instance
                )

                st.subheader("Generated Answer")
                st.info(answer)

                with st.expander("📄 Source Documents (Context Used)"):
                    if documents:
                        for i, doc in enumerate(documents):
                            st.markdown(f"**Document {i+1}** (Source: `{doc.metadata.get('source_file', 'N/A')}`)")
                            st.code(doc.page_content[:500] + ".. .", language='text')
                    else:
                        st.write("No source documents available.")
                        
            except Exception as e:
                st.error(f"❌ Failed to generate answer: {str(e)}")
                print(f"DEBUG: Pipeline execution failed: {str(e)}", file=sys.stderr)
