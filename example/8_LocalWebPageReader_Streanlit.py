import streamlit as st

from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

@st.cache_resource(show_spinner=False)
def create_set_llms()->None:
    # Setting LLM
    llm = Ollama(
        model="mistral",
        temperature=0.2,
        request_timeout=180
    )

    embed = OllamaEmbedding(
        model_name="mxbai-embed-large",
    )

    Settings._llm = llm
    Settings._embed_model = embed
    return None

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode

@st.cache_resource(show_spinner=False)
def create_index()->VectorStoreIndex:
    # Create DB
    chroma_client = chromadb.PersistentClient(path="D:/0_GIT/LLM_Tutorials/example/data/6_chroma")
    chroma_collections = chroma_client.get_collection(name="chromadb")
    chroma_store = ChromaVectorStore(chroma_collection=chroma_collections)

    # Creating index
    index = VectorStoreIndex.from_vector_store(vector_store=chroma_store)
    return index

create_set_llms()
index = create_index()

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, verbose=True)

st.set_page_config(
    page_title="Learn LlamaIndex",
    layout="centered"
)

st.title("Chat with llama-index docs")

if "messages" not in st.session_state.keys():
    data = {}
    data["role"] = "assistant"
    data["content"] = "Hi, welcome to chat & learn"
    st.session_state.messages = []
    st.session_state.messages.append(data)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Your question goes here...")
if prompt:    
    data = {}
    data["role"] = "user"
    data["content"]=prompt
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append(data)

    with st.chat_message("assistant"):        
        response = st.session_state.chat_engine.chat(prompt)    
        st.write(response.response)        
        data = {}
        data["role"] = "assistant"
        data["content"] = response.response
        st.session_state.messages.append(data)        
