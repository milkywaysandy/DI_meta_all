import streamlit as st
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# --- Configuration & Initialization ---

# 1. Load API Key (Important for deployment)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it as an environment variable or secret.")
    st.stop()

# Use st.cache_resource to load the vector store and model once
# This is crucial for performance and cost saving in Streamlit
@st.cache_resource
def initialize_rag_components(api_key):
    """Initializes the embeddings model, vector store, and LLM."""
    try:
        # Initializing Embeddings Model
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=api_key
        )
        
        # NOTE: You MUST ensure 'faiss_index_medical' is available in your deployment package!
        vectorstore_path = "faiss_index_medical" 
        vectorstore = FAISS.load_local(
            folder_path=vectorstore_path,
            embeddings=embeddings_model,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Initializing Generative Model
        genai.configure(api_key=api_key)
        llm_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={"temperature": 0}
        )
        return retriever, llm_model
    except Exception as e:
        st.error(f"Error during RAG initialization: {e}")
        st.stop()

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
    base_prompt = """You are a medical information assistant analyzing drug package inserts... [rest of your prompt]...
    Context:
    {context}

    ANSWER REQUIREMENTS:
    - Be precise and cite specific drug names for each piece of information
    - Use Traditional Chinese in your response
    - If comparing multiple drugs or variants, organize your answer clearly by drug name
    - If information is missing for a specific drug, state this clearly
    - For safety-critical information (contraindications, interactions, dosage), be extra thorough
    - Demonstrate your reasoning by citing the chunks (say their header names instead of chunks if you can) that made your conclusion and why that forms your conclusion

    Question: {question}
    Helpful Answer:"""

    # 4. Inject Context and Question
    final_prompt = base_prompt.format(context=context_text, question=query)

    # 5. Generation
    response = llm_model.generate_content(final_prompt)
    return response.text, most_similar_documents

# --- Streamlit Application Layout ---

st.title("💊 Prototype Drug RAG Assistant")
st.markdown("Enter a medical query to retrieve relevant drug information and generate an answer.")

# Initialize RAG components
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