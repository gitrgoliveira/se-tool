#!/usr/bin/env python3

import logging
import os
import threading
from operator import itemgetter
from typing import Any, ClassVar, Dict, List, Mapping, Optional
from urllib.parse import urlparse

import ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline, EmbeddingsFilter)
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.schema import StrOutputParser, format_document
from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter, EmbeddingsRedundantFilter, LongContextReorder)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler)
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from tqdm import tqdm

import ai.hashi_prompts as hashi_prompts
from ai.embed_hashicorp import get_embedding, output_ai, repo_name
from ai.RAG_sources import repos_and_folders, website_urls

CPU_THREADS = 16
GPU_THREADS = 32
DEFAULT_CACHE_DIR = "./cache"

# pick one from https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
# as long as it exists in https://ollama.ai/library
# llm_model = "mixtral:8x7b"
# llm_model="llama2:13b"
# llm_model = "mixtral:8x7b-instruct-v0.1-q3_K_L"
# llm_model = "notux:8x7b-v1-q3_K_S"
# llm_model = "starling-lm:7b"
default_llm_model = "mistral:7b-instruct-v0.2-q4_0"
# llm_model = "solar:10.7b-instruct-v1-q6_K"

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
            "score_threshold": 0.65
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
    
    # relevant_filter = EmbeddingsFilter(embeddings=filter_embeddings, k=10)

    # filter_ordered_cluster = EmbeddingsClusteringFilter(
    #     embeddings=filter_embeddings,
    #     num_clusters=5,
    #     num_closest=2,
    # )
    # from langchain.retrievers.document_compressors import LLMChainFilter
    # from langchain.retrievers.document_compressors import LLMChainExtractor

    # filter = LLMChainFilter()
    pipeline = DocumentCompressorPipeline(transformers=transformers)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=merger_retriever
    )
    
    return compression_retriever

def simple_search(retrievers, query):
    from langchain.retrievers import EnsembleRetriever
    n_retrievers = len(retrievers)
    weights = [1 / n_retrievers] * n_retrievers
    ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)
    reordering = LongContextReorder()
    return reordering.transform_documents(ensemble_retriever.get_relevant_documents(query))


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

def load_llm(llm_model: str = default_llm_model, host: str = "", callback_manager=None, temperature=0.0) -> Ollama:
    
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
        num_ctx=4096,
        # top_p=0.5,
        top_k=10,
        verbose=True,
        callback_manager=callback_manager,
        )
    
def get_hf_llm():
    from langchain.llms.huggingface_pipeline import HuggingFacePipeline
    hf = HuggingFacePipeline.from_model_id(
        model_id="rishiraj/CatPPT-base",
        task="text-generation",
        device_map="auto",
        model_kwargs={"cache_dir": DEFAULT_CACHE_DIR, "use_fast": True},
        pipeline_kwargs={"temperature": 0},
    )
    return hf

def retrieval_search_chain(llm: Ollama, retriever):
    base_template = hashi_prompts.prompt_from_model(llm.model).format(
        system=("Given the following question, rephrase it as a standalone question, in its original language \n"),
        prompt=("Question: {question} \n",
                "Standalone question:")
    )
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template=base_template)
    standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | get_hf_llm()
    | StrOutputParser(),
    }

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def retrieval_qa_chain(llm: Ollama, retriever: BaseRetriever| None, memory):
    from langchain.chains.conversational_retrieval.prompts import (
        CONDENSE_QUESTION_PROMPT)
    from langchain_core.runnables import (RunnableLambda, RunnableParallel,
                                          RunnablePassthrough)

    
    base_template = hashi_prompts.prompt_from_model(llm.model).format(
        system=("Given the following conversation and a follow up question, rephrase ",
                "the follow up question to be a standalone question, in its original language."),
        prompt=("Chat History: \n",
                "{chat_history}",
                "\nFollow Up Input: {question} \n",
                "Standalone question:")
    )
    # CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template=base_template)
    
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"),
    )
    
    # Calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm #get_hf_llm()
        | StrOutputParser(),
    }
    
    # Retrieve the relevant documents
    if retriever == None:
        retrieved_documents = {
            "docs": {},
            "question": lambda x: x["standalone_question"],
        }

        # Construct the inputs for the final prompt
        final_inputs = {
            "context": {},
            "question": itemgetter("question"),
            "chat_history": lambda x: get_buffer_string(x.get("chat_history", [])),
        }
        
    else:
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | retriever,
            "question": lambda x: x["standalone_question"],
        }
        
        # Construct the inputs for the final prompt
        final_inputs = {
            "context": lambda x: _combine_documents(x["docs"]),
            "question": itemgetter("question"),
            "chat_history": lambda x: get_buffer_string(x.get("chat_history", [])),
        }
    
    # The part that returns the answers
    
    answer = {
        "answer": final_inputs | hashi_prompts.QA_prompt(llm.model) | llm,
        "docs": itemgetter("docs"),
    }
    
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer
        
    
    return final_chain

def get_hashi_search(llm=None, callback_manager=None):
    if llm == None:
        logging.debug("Loading a new LLM")
        loaded_llm = load_llm(callback_manager=callback_manager)
    else:
        loaded_llm=llm
        
    retriever = get_retriever(loaded_llm, use_filters=True)
    
    search = retrieval_search_chain(
        llm=loaded_llm, retriever=retriever
    )
    
    return search

    

def get_hashi_chat(llm=None, callback_manager=None):
    if llm == None:
        logging.debug("Loading a new LLM")
        loaded_llm = load_llm(callback_manager=callback_manager)
    else:
        logging.debug("Using the provided LLM")
        loaded_llm=llm
        
    memory = ConversationSummaryMemory(
        llm=loaded_llm, memory_key="chat_history", return_messages=True
        )
    retriever = get_retriever(loaded_llm, use_filters=True)
    
    qa = retrieval_qa_chain(
        llm=loaded_llm, retriever=retriever, memory=memory
    )
    
    return qa, memory

def check_ollama_host(ollama_host: str) -> str:
    """return url without a final slash and check if valid"""
    url = ollama_host.rstrip("/")
    try:
        urlparse(url)
    except ValueError as e:
        raise ValueError("Invalid OLLAMA_HOST environment variable value") from e
    
    return url

def get_retriever(llm, use_filters=False) -> BaseRetriever | None:
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
    
    from langchain.retrievers.multi_query import MultiQueryRetriever
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=merger_retriever, llm=llm, include_original = True
    )
    retriever = get_pipeline_retriever(retriever_from_llm, embedding_function, use_filters=use_filters)
    
    return retriever


def start_chain():
    qa, memory = get_hashi_chat()
    
    while True:
        print("")
        print("-" * 50)
        print("-" * 50)
        
        query = input("User: ")
        if query == "exit":
            break
        if query == "":
            continue
    
        inputs = {"question": query}
        result = qa.invoke(inputs)
        print(result)
        memory.save_context(inputs, {"answer": result["answer"].content})

        print()
        print("-" * 50)
        print("-" * 50)
        print()
        print("User: ", query)
        print()
        print("AI: ", result["answer"].content)
        print()
        print("Sources: ", )
        if "docs" in result:
            for d in result["docs"]:
                print (d.metadata.get('source', "source not found"))
            print(result)
        else:
            print("No sources found in ", result)



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    start_chain()
    # start_agent()