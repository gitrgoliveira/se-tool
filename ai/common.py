import logging
import os
import threading
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlparse

import ollama
import tqdm
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
from langchain_core.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler)
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from tqdm import tqdm

from ai.embed_hashicorp import get_embedding, output_ai, repo_name
from ai.RAG_sources import repos_and_folders, website_urls

# pick one from https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
# as long as it exists in https://ollama.ai/library
default_llm_model = "mistral:7b-instruct-v0.2-q4_0"


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
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

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
        )

def get_ctx_from_llm(llm_model: str):
    if llm_model.find("mistral"):
        return 32768
    if llm_model.startswith("qwen") or llm_model.startswith("gemma"):
        return 8192
    if llm_model.startswith("phi"):
        return 2048
    
    return 4096
        
    
def get_retriever_tfidf(documents):
    from langchain.retrievers import TFIDFRetriever
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
    bm25_retriever.k = 5
    return bm25_retriever

def get_vectorstore_chroma(persist_directory, embedding_function):
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
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
    reordering = LongContextReorder()
    
    if use_filters:
        filter_dups = EmbeddingsRedundantFilter(embeddings=filter_embeddings)
        transformers.append(filter_dups)
        
        relevant_filter = EmbeddingsFilter(embeddings=filter_embeddings, similarity_threshold=None, k=10)
        transformers.append(relevant_filter)
    
    from flashrank import Ranker

    from ai.flashrank_rerank import FlashrankRerank
    reranker = FlashrankRerank(
        client=Ranker(model_name="rank-T5-flan",
        cache_dir=DEFAULT_CACHE_DIR),
        top_n=5, 
        cache_dir=DEFAULT_CACHE_DIR)
        
    transformers.append(reranker)
    
    transformers.append(reordering)
    
    pipeline = DocumentCompressorPipeline(transformers=transformers)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=merger_retriever
    )
    
    return compression_retriever


def get_filter_embedding():
    host = check_ollama_host(os.getenv('OLLAMA_HOST', "http://localhost:11434"))
    ModelDownloader(host=host).download_model(default_llm_model)    
    embedding = OllamaEmbeddings(
        base_url=host,
        model = default_llm_model,
        # model = "phi:2.7b",
        # model = "starling-lm:7b",
        num_gpu = GPU_THREADS,
        num_thread = CPU_THREADS,
        show_progress = True,
        mirostat = 2,
        num_ctx = 4096,
        temperature=0,
        top_k=10,
        )
    
    return embedding

def get_hf_llm():
    from langchain.llms.huggingface_pipeline import HuggingFacePipeline
    hf = HuggingFacePipeline.from_model_id(
        model_id="rishiraj/CatPPT-base",
        task="text-generation",
        device_map="auto",
        model_kwargs={"cache_dir": DEFAULT_CACHE_DIR},
        pipeline_kwargs={"temperature": 0},
    )
    return hf


def get_retriever(llm, use_filters=False, multi_query=False) -> BaseRetriever | None:
    embedding_function = get_embedding()
    # filter_embedding = get_filter_embedding()
    chroma_retrievers = []
    chroma_retrievers = get_chroma_retrievers(output_ai, embedding_function)
    # chroma_retrievers.append(get_vectorstore_chroma(output_ai, embedding_function))
    
    # documents = load_documents_md(output_md)
    # print("Adding BM25 retriever for diversity.")
    # bm25_retriever = get_retriever_bm25(documents)
    # chroma_retrievers.append(bm25_retriever)
    # tfidf_retriever = get_retriever_tfidf(documents)
    # chroma_retrievers.append(tfidf_retriever)
    logging.info(f"Loaded {len(chroma_retrievers)} retrievers")
    if len(chroma_retrievers) == 0:
        return None
    
    merger_retriever = MergerRetriever(retrievers=chroma_retrievers)
    # retriever = get_pipeline_retriever(merger_retriever, embedding_function, use_filters=use_filters)
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
