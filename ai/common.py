import logging
import os
import threading
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse

import ollama
from flashrank import Ranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline, EmbeddingsFilter)
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_community.document_transformers.embeddings_redundant_filter import (
    EmbeddingsRedundantFilter)
from langchain_community.document_transformers.long_context_reorder import (
    LongContextReorder)
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler)
from langchain_core.documents.base import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from tqdm import tqdm

from ai.embed_hashicorp import get_embedding, output_ai, repo_name, split_text
from ai.flashrank_rerank import FlashrankRerank
from ai.RAG_sources import repos_and_folders, website_urls

# pick one from https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
# as long as it exists in https://ollama.ai/library
default_llm_model = "mistral:7b"
# base folder for adhoc documents 
extra_files = "docs"

CPU_THREADS = 16
GPU_THREADS = 32
DEFAULT_CACHE_DIR = "./cache"

class ModelDownloader:
    _instance = None
    download_lock = threading.Lock()
    cli : ollama.Client
    
    def __new__(cls, host: str | None, *args, **kwargs):
        if not cls._instance:
            if host == "" or host == None:
                host = os.getenv('OLLAMA_HOST', "http://localhost:11434")
                host = check_ollama_host(host)
            cls.cli = ollama.Client(host=host)
            cls._instance = super(ModelDownloader, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    @classmethod
    def download_model(cls, llm_model: str, force: bool = False) -> bool:
        with cls.download_lock:            
            if (not cls.model_exists(llm_model)) or force:
                logging.info(f"Downloading model {llm_model}")
                progress_response = cls.cli.pull(model=llm_model, stream=True)
    
                if isinstance(progress_response, ollama.RequestError):
                    logging.error(f"Failed to download file: {progress_response}")
                    return False
                    
                # Initialize the progress bar
                pbar: Optional[tqdm] = None
                for line in progress_response:
                    if line:
                        # Parse the JSON string into a Python dictionary
                        line_dict: Dict[str, Any] = dict(line)

                        # Update the progress bar based on the 'completed' and 'total' fields
                        if 'total' in line_dict and 'completed' in line_dict:
                            if pbar is None:
                                # Create the progress bar
                                pbar = tqdm(total=line_dict['total'], unit='B', unit_scale=True)
                            pbar.update(line_dict['completed'] - pbar.n)

                if pbar is not None:
                    pbar.close()

            else:
                logging.warning(f"Model already downloaded {llm_model}")
        return True

    @classmethod
    def model_exists(cls, llm_model: str) -> bool:
        models = cls.cli.list()
        return any(model['name'] == llm_model for model in models['models'])
    
    @classmethod
    def list(cls) -> Mapping[str, Any]:
        return cls.cli.list() 
 
 
 
def check_ollama_host(ollama_host: str) -> str:
    """return url without a final slash and check if valid"""
    url = ollama_host.rstrip("/")
    try:
        urlparse(url)
    except ValueError as e:
        raise ValueError("Invalid OLLAMA_HOST environment variable value") from e
    
    return url



def load_llm(llm_model: str = default_llm_model,
             host: str = "",
             callback_manager=None,
             temperature=0.0) -> Ollama:
    
    if callback_manager is None:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler(), StdOutCallbackHandler()])

    ollama_host = host
    if ollama_host == "":
        ollama_host = os.getenv('OLLAMA_HOST', "http://localhost:11434")

    ollama_host = check_ollama_host(ollama_host)
    logging.info(f"Loaded Ollama from {ollama_host}")
    ModelDownloader(host=ollama_host).download_model(llm_model)
    
    return Ollama(
        base_url=ollama_host,
        model=llm_model,
        mirostat=2,
        # num_gpu=GPU_THREADS,
        # num_thread=CPU_THREADS,
        temperature=temperature,
        num_ctx=get_ctx_from_llm(llm_model),
        # top_p=0.5,
        top_k=10,
        verbose=True,
        callback_manager=callback_manager,
        keep_alive="25m"
        )

def get_ctx_from_llm(llm_model: str):
    if llm_model.startswith("mistral"):
        return 32768
    if llm_model.startswith("qwen") \
        or llm_model.startswith("gemma") \
            or llm_model.startswith("llama3"):
        return 8192
    if llm_model.startswith("phi"):
        return 2048
    if llm_model.startswith("command-r"):
        return 131072
    
    return 4096
        
def get_retriever_svm (documents, embeding_function):
    from langchain_community.retrievers.svm import SVMRetriever
    retriever = SVMRetriever.from_documents(documents=documents, embeddings=embeding_function)
    retriever.relevancy_threshold = 0.55
    retriever.k = 10
    
    return retriever

def get_retriever_parent (documents, embeding_function):
    from langchain.retrievers import ParentDocumentRetriever
    from langchain_community.vectorstores import InMemoryVectorStore

    from langchain.storage import InMemoryStore
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=512)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2048)

    vectorstore = InMemoryVectorStore(embedding=embeding_function)
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    retriever.add_documents(documents)
    
    return retriever
    
def get_retriever_tfidf(documents):
    from langchain_community.retrievers.tfidf import TFIDFRetriever
    tfidf_retriever = TFIDFRetriever.from_documents(
        documents=documents,
        tfidf_params={
            'encoding' : 'utf-16',
            'decode_error': 'replace',
            'lowercase': False,
            
        }                                         
        )
    tfidf_retriever.k = 3
    return tfidf_retriever

def get_retriever_bm25(documents):
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10
    return bm25_retriever

def get_vectorstore_chroma(persist_directory, embedding_function):
    # from chromadb.config import Settings
    # client_settings = Settings()
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    vectorstore._client_settings.anonymized_telemetry = False
    # vectorstore._client_settings.chroma_product_telemetry_impl = ""
    # vectorstore._client_settings.chroma_telemetry_impl = ""
    return vectorstore

def get_retriever_chroma(vectorstore: VectorStore):
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 10,
            "score_threshold": 0.55
            }
        
        # search_type="mmr", # this makes all vector stores return results regardless of quality
        # search_kwargs = {
        #     'fetch_k': 30, 
        #     "k": 10,
        #     "lambda_mult": 1,
        # },
        
    )
    
    return retriever

def get_chroma_retrievers(path, embedding_function):
    
    retrievers = []
    
    for repo_info in repos_and_folders:
        name = repo_name(repo_info["repo_url"])
        repo_path = os.path.join(path, "git", name)
        if os.path.exists(repo_path):
            logging.debug(f"Loading git embeddings for {name}")
            vectorstore = get_vectorstore_chroma(repo_path, embedding_function)
            retrievers.append(get_retriever_chroma(vectorstore))

    for url in website_urls:
        repo_path = os.path.join(path, "web", url['name'])
        if os.path.exists(repo_path):
            logging.debug(f"Loading web embeddings for {url}")
            vectorstore = get_vectorstore_chroma(repo_path, embedding_function)
            retrievers.append(get_retriever_chroma(vectorstore))
            
    return retrievers


def get_pipeline_retriever(merger_retriever, filter_embeddings, use_filters=False) -> BaseRetriever:
    transformers = []
    
    if use_filters:
        filter_dups = EmbeddingsRedundantFilter(embeddings=filter_embeddings)
        transformers.append(filter_dups)
        
        relevant_filter = EmbeddingsFilter(embeddings=filter_embeddings, similarity_threshold=None, k=10)
        transformers.append(relevant_filter)
            
    transformers.append(FlashrankRerank(
        client=Ranker(model_name="rank-T5-flan",
        cache_dir=DEFAULT_CACHE_DIR),
        top_n=6, 
        cache_dir=DEFAULT_CACHE_DIR)
    )
    
    transformers.append(LongContextReorder())
    
    pipeline = DocumentCompressorPipeline(transformers=transformers)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=merger_retriever
    )
    
    return compression_retriever


def get_filter_embedding():
    host = check_ollama_host(os.getenv('OLLAMA_HOST', "http://localhost:11434"))
    # emebedding_model = "nomic-embed-text:v1.5" 
    emebedding_model = "snowflake-arctic-embed:335m"
    ModelDownloader(host=host).download_model(emebedding_model)    
    embedding = OllamaEmbeddings(
        base_url=host,
        model = default_llm_model,
        num_gpu = GPU_THREADS,
        num_thread = CPU_THREADS,
        show_progress = True,
        mirostat = 2,
        # num_ctx = 4096,
        temperature=0,
        # top_k=10,
        )
    
    return embedding

def get_hf_llm():
    from langchain_community.llms import HuggingFacePipeline
    hf = HuggingFacePipeline.from_model_id(
        model_id="rishiraj/CatPPT-base",
        task="text-generation",
        device_map="auto",
        model_kwargs={"cache_dir": DEFAULT_CACHE_DIR},
        pipeline_kwargs={"temperature": 0},
    )
    return hf


def get_retriever(llm, use_filters=False, multi_query=False, extra_retriever: BaseRetriever = None) -> BaseRetriever | None:
    embedding_function = get_embedding()
    # filter_embedding = get_filter_embedding()
    chroma_retrievers = []
    chroma_retrievers = get_chroma_retrievers(output_ai, embedding_function)
    
    if extra_retriever is not None:
        chroma_retrievers.append(extra_retriever)
        
    logging.info(f"Loaded {len(chroma_retrievers)} retrievers")
    if len(chroma_retrievers) == 0:
        return None
    
    merger_retriever = MergerRetriever(retrievers=chroma_retrievers)

    if multi_query:
        from langchain.retrievers.multi_query import MultiQueryRetriever
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=merger_retriever, llm=llm, include_original = True
        )
        retriever = get_pipeline_retriever(retriever_from_llm, embedding_function, use_filters=use_filters)
    else:
        retriever = get_pipeline_retriever(merger_retriever, embedding_function, use_filters=use_filters)
        
    return retriever

def load_extra_retriever() -> BaseRetriever:
    embedding_function = get_embedding()
    documents = load_extra_files(extra_files)
    if documents == []:
        return None
    
    logging.info(f"Adding SVM retriever with extra documents.")
    # retriever = get_retriever_svm(documents, embedding_function)
    retriever = get_retriever_parent(documents, embedding_function)
    
    return retriever

# returns an array of documents from the extra_files folder
def load_extra_files(path):
    if not os.path.exists(path):
        return []
    
    extra_documents = []
    for file in os.listdir(path):
        logging.info(f"Loading {file}")
        if file.endswith(".pdf"):
            extra_documents.extend(load_and_split_pdf(os.path.join(path, file)))
        elif file.endswith(".docx") or file.endswith(".doc"):
            extra_documents.extend(load_and_split_docx(os.path.join(path, file)))
        elif file.endswith(".xlsx") or file.endswith(".xls") or file.endswith(".xlsm"):
            extra_documents.extend(load_and_split_xlsx(os.path.join(path, file)))
        elif file.endswith(".pptx") or file.endswith(".ppt"):
            extra_documents.extend(load_and_split_pptx(os.path.join(path, file)))
        elif file.endswith(".csv"):
            extra_documents.extend(load_and_split_csv(os.path.join(path, file)))
        
            
    return extra_documents

def load_and_split_pdf(path) -> List[Document]:
    try: 
        from langchain_community.document_loaders.pdf import PyPDFLoader
        loader = PyPDFLoader(path, extract_images=False)
        docs = loader.load()
        # return split_text(docs)
        return docs
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
    return []

def load_and_split_docx(path) -> List[Document]:
    try:
        from langchain_community.document_loaders.word_document import (
            UnstructuredWordDocumentLoader)
        loader = UnstructuredWordDocumentLoader(path)
        docs = loader.load()
        # return split_text(docs)
        return docs
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
    return []

def load_and_split_xlsx(path) -> List[Document]:
    try:
        from langchain_community.document_loaders.excel import (
            UnstructuredExcelLoader)
        loader = UnstructuredExcelLoader(path, mode="elements")
        docs = loader.load()
        # return split_text(docs)
        return docs
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
    return []

def load_and_split_pptx(path) -> List[Document]:
    try:
        from langchain_community.document_loaders.powerpoint import (
            UnstructuredPowerPointLoader)
        loader = UnstructuredPowerPointLoader(path)
        docs = loader.load()
        # return split_text(docs)
        return docs
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
    return []


def load_and_split_csv(path) -> List[Document]:
    try:
        from langchain_community.document_loaders.csv_loader import CSVLoader
        loader = CSVLoader(file_path=path, autodetect_encoding=True)
        return loader.load()
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
    return []
