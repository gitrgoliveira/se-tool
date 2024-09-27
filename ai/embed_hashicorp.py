#!/usr/bin/env python3

import datetime
import os
import sys
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from logging import debug, error, info
from typing import List

import torch
from dateutil.relativedelta import relativedelta
from langchain.indexes import SQLRecordManager, index
from langchain_chroma import Chroma
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_transformers.html2text import (
    Html2TextTransformer)
from langchain_community.embeddings.huggingface import (  # HuggingFaceEmbeddings,
    HuggingFaceBgeEmbeddings, HuggingFaceInstructEmbeddings)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents.base import Document
from langchain_text_splitters import (MarkdownTextSplitter, NLTKTextSplitter,
                                      SentenceTransformersTokenTextSplitter)

from ai.RAG_sources import repos_and_folders, website_urls

CPU_THREADS = 16
GPU_THREADS = 32

# base folder for the generated database files
output_ai = "output_ai_files"
# base folder where the cloned repos will be stored
output_repos = "output_cloned_repos"

# embeddings leaderboard: https://huggingface.co/models?pipeline_tag=embeddings&sort=downloads
# Massive Text Embedding Benchmark (MTEB) leaderboard: https://huggingface.co/spaces/mteb/leaderboard
model_name = "BAAI/bge-large-en-v1.5"
tokens_per_chunk = 512

# For when there's enough memory to run mistral embeddings
# model_name = "intfloat/e5-mistral-7b-instruct"
# tokens_per_chunk = 4096

def repo_name(repo_url):
    repo_name = repo_owner_and_name(repo_url)
    return repo_name.replace("/", "_")

def repo_owner_and_name(repo_url):
    return "/".join(repo_url.split("/")[-2:]).split(".git")[0]

class JobProcessor:
    def __init__(self, thread_prefix="embed", max_workers=5):
        self.executor = ThreadPoolExecutor(thread_name_prefix=thread_prefix, max_workers=max_workers)

    def submit_job(self, func, *args, **kwargs):
        return self.executor.submit(func, *args, **kwargs)

    def shutdown(self):
        self.executor.shutdown(wait=True)

load_processor = JobProcessor(max_workers=3, thread_prefix="loader")
embedding_processor = JobProcessor(max_workers=2, thread_prefix="embedder")


def load_documents_md(data_path):
    info(f"Loading data from {data_path}...") 
    loader = DirectoryLoader(data_path, glob="**/*", recursive=True, show_progress=True, 
                             use_multithreading=True, 
                             max_concurrency=CPU_THREADS,
                             loader_cls=TextLoader, loader_kwargs={'autodetect_encoding': True},
                             silent_errors=True)
    

    chunks = loader.load()
    info(f"Loaded {len(chunks)} documents")
    
    return splitter_md (chunks)

def splitter_md (chunks):
    md_splitter = MarkdownTextSplitter(keep_separator=True, add_start_index=True)
    html2text = Html2TextTransformer(ignore_images=False, ignore_links=False)
    info("Splitting...")
    chunks = md_splitter.split_documents(chunks)
    info(f"Split into {len(chunks)} chunks, using Markdown Splitter")
    
    chunks = html2text.transform_documents(chunks)
    info(f"Converted HTML to Text on {len(chunks)} chunks")
    
    return split_text(chunks)

def split_text(chunks) -> List[Document]:
    
    debug("Filtering...")
    chunks = filter_complex_metadata(chunks)
    info(f"Filtered into {len(chunks)} chunks")
    
    chunks = nltk_splitter(chunks)
    # chunks = semantic_splitter(chunks) # this is currently very slow to run locally!
    chunks = stt_splitter(chunks)
    
    return chunks

def nltk_splitter(chunks) -> List[Document]:    
    nltk_splitter = NLTKTextSplitter(add_start_index=True)
    chunks = nltk_splitter.split_documents(chunks)
    info(f"Split into {len(chunks)} chunks, using NLTK Text Splitter")
    
    return chunks

# this is currently very slow to run locally!
def semantic_splitter(chunks) -> List[Document]:
    from langchain_experimental.text_splitter import SemanticChunker
    hf_embeddings = get_embedding()
    semantic_chunker = SemanticChunker(
                                    embeddings=hf_embeddings,
                                    add_start_index=True,
                                    breakpoint_threshold_type="gradient",   
                                    )
    
    chunks = semantic_chunker.split_documents(chunks)
    info(f"Split into {len(chunks)} chunks, using the Semantic Chunker")
    
    return chunks
    
    
def stt_splitter(chunks) -> List[Document]:
    info(f"Splitting text into chunks of {tokens_per_chunk} tokens")
    
    stt_splitter = SentenceTransformersTokenTextSplitter(
        model_name = model_name, 
        tokens_per_chunk = tokens_per_chunk,
        add_start_index=True,
        keep_separator = True
        )
    stt_splitter._add_start_index = True
    
    chunks = stt_splitter.split_documents(chunks)
    info(f"Split into {len(chunks)} chunks, using Sentence Transformers Token Text Splitter")
    
    return chunks

def load_documents_git(repo_path, repo_url=None):
    info(f"Loading data from {repo_path}...")

    from langchain_community.document_loaders import GitHubIssuesLoader
    chunks = []
    
    # load only issues from the 3 years
    three_years_ago = datetime.datetime.now() - relativedelta(years=3)
    since = three_years_ago.isoformat(timespec="seconds")+'Z'
    
    github_loader = GitHubIssuesLoader(access_token=os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", ""),
                                       repo=repo_owner_and_name(repo_url),
                                       since=since)
    
    try:
        more_chunks = github_loader.load()
        info(f"Loaded {len(more_chunks)} issues")
        more_chunks = splitter_md (more_chunks)
        # from langchain_community.document_transformers.doctran_text_qa import DoctranQATransformer
        # qa_docs = DoctranQATransformer().transform_documents(more_chunks)
        # chunks.extend(qa_docs)
        chunks.extend(more_chunks)
        
    except Exception as e:
        error(f"Error loading issues {e}")
        
    return chunks

  
def get_embedding():
    
    model_kwargs = {'device': 'cpu'}
    if torch.cuda.is_available():
        model_kwargs = {'device': 'cuda'}
    elif sys.platform == "darwin":
        model_kwargs = {'device':'mps'}
        
    encode_kwargs = {'normalize_embeddings': False}
    cache_folder="./cache"
    os.makedirs(cache_folder, exist_ok=True)
    return HuggingFaceBgeEmbeddings(cache_folder=cache_folder,
                                    model_name=model_name,
                                    model_kwargs=model_kwargs,
                                    encode_kwargs = encode_kwargs,
                                    show_progress = True
    ) 

def get_embedding_mistral():
    
    model_kwargs = {'device': 'cpu'}
    if torch.cuda.is_available():
        model_kwargs = {'device': 'cuda'}
    elif sys.platform == "darwin":
        model_kwargs = {'device':'mps'}

    encode_kwargs = {'normalize_embeddings': False}
    # model_name = "intfloat/e5-mistral-7b-instruct",
    # tokens_per_chunk = 4096                     
    # For when there's enough memory to run mistral embeddings
    cache_folder="./cache"
    os.makedirs(cache_folder, exist_ok=True)

    return HuggingFaceInstructEmbeddings(cache_folder=cache_folder,
                                        model_name="intfloat/e5-mistral-7b-instruct",
                                        model_kwargs=model_kwargs,
                                        encode_kwargs = encode_kwargs,
                                        show_progress = True
    )

def create_embeddings(name, documents, output_path):
    info(f"Creating {name} embeddings for {len(documents)} documents")
    info(f"Output path {output_path}")
    hf_bge_embeddings = get_embedding()
    record_manager = SQLRecordManager(
        name, db_url=f"sqlite:///{output_path}/record_manager_cache.sqlite"
    )
    record_manager.create_schema()
    
    vectorstore = Chroma(embedding_function=hf_bge_embeddings, persist_directory=output_path)
    # def _clear():
    #     """Hacky helper method to clear content. See the `full` mode section to to understand why it works."""
    #     index([], record_manager, vectorstore, cleanup="full", source_id_key="source")

    # _clear()
    result = index(documents, record_manager, vectorstore, cleanup="full", source_id_key="source")
    
    info(f"Results: {result}")
    # writes the results in a ".done" file with the same folder as confirmation.
    with open(f"{output_path}/results.done", "w") as file:
        file.write(str(result))
    
    # vectorstore = Chroma.from_documents(documents=documents,
    #                                     embedding=hf_bge_embeddings,
    #                                     persist_directory=output_path)
    # vectorstore.persist()

    # info(f"Created embeddings for {len(documents)} documents")
    
def add_repo_metadata(docs, repo_url):
    info(f"Adding repo metadata to {len(docs)} documents")
    for d in docs:
        if 'url' in d.metadata:
            d.metadata['source'] = d.metadata['url']
        elif 'source' in d.metadata:
            d.metadata['source'] = str(repo_url).split(".git")[0] +"/blob/main/"+d.metadata['source']
        else:
            d.metadata['source'] = repo_url
            
    return docs

def recursive_website_loader(url: dict):
    info(f"Loading data from {url}")

    html2text = Html2TextTransformer(ignore_images=False, ignore_links=False)

    from ai.web_scraper import Scraper
    docs = Scraper(base_url=url['url'], 
                   max_depth=url['depth'], 
                   prevent_outside=url['prevent_outside'],
                   debug=True
                   ).load_and_split(NLTKTextSplitter(add_start_index=True))
    docs = html2text.transform_documents(docs)
    
    info(f"Loaded {len(docs)} documents from {url}")    
    return docs

def check_github_token():
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
    if token is None:
        error("GITHUB_PERSONAL_ACCESS_TOKEN not set.")
    return token

def create_git_embeddings(base_path: str, only_missing_embeddings: bool):
    if check_github_token() is None:
        return
    
    jobs = []
    for repo_info in repos_and_folders:
        name = repo_name(repo_info["repo_url"])
        repo_path = os.path.join(output_repos, name)
        output_path = os.path.join(base_path, name)
        if only_missing_embeddings and os.path.exists(os.path.join(output_path, "results.done")):
            continue
        future = load_processor.submit_job(load_documents_git, repo_path, repo_info["repo_url"])
        jobs.append((repo_info["repo_url"], output_path, future))
    
    # Process the jobs and collect the results
    for repo_url, output_path, future in jobs:
        name = repo_name(repo_url)
        github_docs = future.result()
        info(f"Loaded {len(github_docs)} documents from {name}")
        if len(github_docs) == 0:
            continue
        docs = add_repo_metadata(github_docs, repo_url)
        os.makedirs(output_path, exist_ok=True)
        embedding_processor.submit_job(create_embeddings, name, docs, output_path)
    

from typing import List, Dict, Tuple

def create_website_embeddings(base_path: str, only_missing_embeddings: bool):
    jobs: List[Tuple[str, str, Future]] = []
    
    for website_url in website_urls:
        output_path = os.path.join(base_path, website_url['name'])
        if only_missing_embeddings and os.path.exists(os.path.join(output_path, "results.done")):
            continue
        future = load_processor.submit_job(recursive_website_loader, website_url)
        jobs.append((website_url['name'], output_path, future))
    
    # as scrapping jobs complete, we need to start the embedding
    for future in as_completed([job[2] for job in jobs]):
        job = next(job for job in jobs if job[2] == future)
        name, output_path, _ = job
        try:
            website_docs = future.result()
            info(f"Loaded {len(website_docs)} documents from {name}")
            if len(website_docs) == 0:
                continue
            embedding_processor.submit_job(web_split_and_embed, name, output_path, website_docs)
        except Exception as e:
            error(f"Error processing {name}: {e}")


def web_split_and_embed(name, output_path, website_docs):
    website_docs = split_text(website_docs)
    os.makedirs(output_path, exist_ok=True)
    create_embeddings(name, website_docs, output_path)

def wait_until_finished():
    load_processor.shutdown()
    embedding_processor.shutdown()