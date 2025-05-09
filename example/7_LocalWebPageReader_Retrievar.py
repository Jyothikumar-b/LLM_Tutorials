## Add debugging cabability
from llama_index.core.callbacks import LlamaDebugHandler
from llama_index.core.callbacks import CallbackManager
from llama_index.core.settings import Settings

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

Settings._callback_manager = callback_manager

## Setting LLM
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

llm = Ollama(
    model="mistral",
    temperature=0.2
)

embed = OllamaEmbedding(
    model_name="mxbai-embed-large",
)

Settings._llm = llm
Settings._embed_model = embed

## Preparing DB
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex

chroma_client = chromadb.PersistentClient(path="./example/data/6_chroma")
chroma_collections = chroma_client.get_collection(name="chromadb")
chroma_store = ChromaVectorStore(chroma_collection=chroma_collections)

### Creating index
index = VectorStoreIndex.from_vector_store(vector_store=chroma_store)

### Engine
engine = index.as_query_engine()

### Check
res = engine.query("What is llama index?")
print(res.response)

### Sample output
# **********
# Trace: index_construction
# **********
# **********
# Trace: query
#     |_CBEventType.QUERY -> 25.10392 seconds
#       |_CBEventType.RETRIEVE -> 4.601024 seconds
#         |_CBEventType.EMBEDDING -> 4.524626 seconds
#       |_CBEventType.SYNTHESIZE -> 20.502896 seconds
#         |_CBEventType.TEMPLATING -> 0.0 seconds
#         |_CBEventType.LLM -> 20.496342 seconds
# **********
#  LlamaIndex appears to be a toolkit that facilitates the creation of both steps for managing data and generating responses. It offers various components such as Data Connectors, Documents/Nodes, Data Indexes, and modules for building RAG pipelines for Q&A, chatbots, or agents. It also supports integration with Managed Indices like VectaraIndex.
