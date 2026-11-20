import os
import shutil
from tkinter.ttk import Style
from turtle import width
from typing import Iterator
from agno.agent import Agent
from agno.run.agent import  RunOutputEvent
from agno.utils.pprint import pprint_run_response
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder

# from agno.knowledge.embedder.openai import OpenAIEmbedder
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough


# from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAIChat
from agno.vectordb.lancedb import LanceDb, SearchType
# from langchain_groq import ChatGroq
from agno.models.groq import Groq
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
import base64

from phoenix.otel import register

# Set environment variables for Arize Phoenix
# os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('ARIZE_PHOENIX_API_KEY')}"
os.environ["PHOENIX_API_KEY"] = os.getenv('ARIZE_PHOENIX_API_KEY')
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com/s/harryaigemini"

# Configure the Phoenix tracer
tracer_provider = register(
    project_name="default", 
    auto_instrument=True,  # Automatically use the installed OpenInference instrumentation
)


st.set_page_config(page_title="Agentic RAG - Harry", layout="wide")

tracer = tracer_provider.get_tracer(__name__)

@tracer.chain
def load_knowledge_base(urls: list[str] = None):
    """
    Returns the knowledge base for the agent.
    This function is used to load the knowledge base from a URL.
    """
    knowledge_base = Knowledge(
        # urls=urls or [],
        vector_db=LanceDb(
            table_name="mcp-docs-knowledge-base",
            uri="tmp/lancedb",
            search_type=SearchType.vector,
            embedder= SentenceTransformerEmbedder(
                    id="sentence-transformers/all-MiniLM-L6-v2",
                    dimensions=384, # Optional: auto-detected for common models
                ), #OpenAIEmbedder(id="text-embedding-3-small"),
        ),
    )
    # knowledge_base.load()
    knowledge_base.insert(urls)
    return knowledge_base


# Load the knowledge base: Comment after first run as the knowledge base is already loaded
@tracer.chain
def agentic_rag_response(
    urls: list[str] = None, query: str = ""
) -> Iterator[RunOutputEvent]:
    knowledge_base = load_knowledge_base(urls)

    agent = Agent(
        # model=OpenAIChat(id="gpt-5-2025-08-07"),
        model= Groq(
            id="openai/gpt-oss-120b",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        ),
        knowledge=knowledge_base,
        search_knowledge=True,
        # show_tool_calls=True,
        markdown=True,
    )

    response: Iterator[RunOutputEvent] = agent.run(query, stream=True)

    return response


col1, col2 = st.columns([4, 1])
with col1:
    title_html = f"""
        <div style="display: flex; align-items: center; gap: 10px;">
            <h1 style="margin: 0;">
             Agentic RAG with Agno & GPT-5
            </h1>
        </div>
        """
    st.markdown(title_html, unsafe_allow_html=True)

with col2:
        if st.button("🔄 Reset KB"):
            st.session_state.docs_loaded = False
            if 'loaded_urls' in st.session_state:
                del st.session_state['loaded_urls']
            st.success("Knowledge base reset!")
            st.rerun()

with st.sidebar:
    st.markdown("### 🧠 Knowledge Base URLs")
    if "urls" not in st.session_state:
        st.session_state.urls = [""]

    col1, col2 = st.columns([4, 1])
    with col1:
        for i, url in enumerate(st.session_state.urls):
            st.session_state.urls[i] = col1.text_input(
                f"URL {i+1}", value=url, key=f"url_{i}", label_visibility="collapsed"
            )
    # Add button in the last column
    if col2.button("➕"):
        if st.session_state.urls and st.session_state.urls[-1].strip() != "":
            st.session_state.urls.append("")

    # Remove empty strings and duplicates
    urls = [u for u in st.session_state.urls if u.strip()]
    urls = list(dict.fromkeys(urls))  # Remove duplicates, preserve order

    if st.button("Load Knowledge Base"):
        if urls:
            with st.spinner("Loading knowledge base... This may take a moment."):
                try:
                    # Actually load the knowledge base with the provided URLs
                    knowledge_base = load_knowledge_base(urls)
                    st.session_state.docs_loaded = True
                    st.session_state.loaded_urls = urls.copy()  # Store the loaded URLs
                    st.success(f"Knowledge base loaded successfully with {len(urls)} URL(s)!")
                except Exception as e:
                    st.error(f"Error loading knowledge base: {str(e)}")
                    st.session_state.docs_loaded = False
        else:
            st.warning("Please add at least one URL to the knowledge base.")
    
    # Display currently loaded URLs if any
    if st.session_state.get('docs_loaded', False) and st.session_state.get('loaded_urls'):
        st.markdown("**📚 Currently Loaded URLs:**")
        for i, url in enumerate(st.session_state.loaded_urls, 1):
            st.markdown(f"{i}. {url}")
    
    st.markdown("---")

query = st.chat_input("Ask a question", width=1000)
if query:
    # Check if knowledge base is loaded
    if not st.session_state.get('docs_loaded', False):
        st.warning(
            "Please load the knowledge base first by adding URLs and clicking 'Load Knowledge Base'."
        )
    elif not st.session_state.get('loaded_urls'):
        st.warning(
            "No URLs are currently loaded in the knowledge base. Please add URLs and load the knowledge base."
        )
    else:
        # Use the loaded URLs from session state
        loaded_urls = st.session_state.loaded_urls
        response = agentic_rag_response(loaded_urls, query)
        st.markdown("#### Answer", unsafe_allow_html=True)
        answer = ""
        answer_placeholder = st.empty()
        for content in response:
            #print(f"{content.event}")
            if hasattr(content, 'event') and content.event == "RunContent":
                #print(f"in IF \n{content.content}")
                answer += content.content
                answer_placeholder.markdown(answer, unsafe_allow_html=True)


# if __name__ == "__main__":
# response = agentic_rag_response(["https://modelcontextprotocol.io/docs/learn/architecture.md"], "Tell me about MCP primitives that clients can expose.")
# pprint_run_response(response, markdown=True)
