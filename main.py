import os
from typing import List

import ollama
import streamlit as st

from ai.common import (ModelDownloader, check_ollama_host,
                       load_extra_retriever, load_llm)
from ai.hashi_chat import get_hashi_chat
from ai.hashi_search import get_hashi_search
from ui.streamlit_assistant import hashi_assistant
from ui.streamlit_chat import hashi_chat
from ui.streamlit_playground import add_playground
from ui.streamlit_writer import hashi_writer


def get_model_list(ollama_host: str | None) -> List[str]:
    try:
        if ollama_host != None:
            ollama_host = check_ollama_host(ollama_host=ollama_host)
        cli = ollama.Client(host=ollama_host)
        models = cli.list()
    except Exception as e:
        st.error("Error loading Ollama model list {e}".format(e=e))
        return []
    list_models = [model['name'] for model in models['models']]
    list_models.sort()
    return list_models
    
# Create the Streamlit app
def main():
    st.set_page_config(
        page_title="HashiCorp SE AI Tools",
        layout="wide")
    
    st.title("Tools to Search, Research, and Write")
    with st.sidebar:
        settings()
    
    writer, chat, search, assistant, playground = st.tabs(["Write", "Chat", "Search", "Research", "AI Playground"])    
    with search:
        hashi_assistant()
    with assistant:
        st.write("Coming soon!")
    with writer:
        hashi_writer()
    with chat:
        hashi_chat()
    with playground:
        add_playground()

# @st.fragment()
def settings():
    st.title("Ollama Settings")
    default_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_host = st.text_input("OLLAMA_HOST", key="OLLAMA_HOST", value=default_host, placeholder="OLLAMA_HOST", label_visibility="collapsed")
    ollama_host = check_ollama_host (ollama_host=ollama_host)
        
    col1, col2 = st.columns([3,1])
    with col1:
        pull_model = st.text_input("Pull model", placeholder="model to pull", label_visibility="collapsed")
    with col2:
        pull_button = st.button("Pull", type="secondary", use_container_width=True)
        
    if pull_button:
        try:
            with st.spinner("Downloading model..."):
                ModelDownloader(host=ollama_host).download_model(pull_model)
        except Exception as ex:
            st.error(f"{ex}")
        
    if get_model_list(ollama_host=ollama_host) == []:
        st.error("No models found. Please run `ollama pull` to download models from https://ollama.ai/library")
        
    model_list = get_model_list(ollama_host=ollama_host)
    index = 0
    if 'llm_model' in st.session_state and model_list != []:
        index = model_list.index(st.session_state['llm_model'])
            
    llm_model = st.selectbox("Select the language model",
                                     options=model_list,
                                     label_visibility="collapsed",
                                     index=index)

    temperature = st.slider("Less or more creative?", min_value=0.0, max_value=1.0, value=0.1, step=.1)
    load_llm_button = st.button("Load model", use_container_width=True, type="primary")
    reload_extra_docs_disabled = False
    if st.session_state.get('llm', None) == None or load_llm_button:
        reload_extra_docs_disabled = True
    reload_extra_docs = st.button("Reload extra docs", use_container_width=True, type="secondary", disabled=reload_extra_docs_disabled)    
    
    if reload_extra_docs:
        with st.spinner("Reloading extra docs..."):
            st.session_state['extra_retriever'] = load_extra_retriever()
            st.toast("Reloaded extra docs")
    
    if load_llm_button or reload_extra_docs:
        with st.spinner("Loading language model..."):
            if 'search' in st.session_state:
                del st.session_state['search']
            if 'chat' in st.session_state:
                del st.session_state['chat']
            if 'llm' in st.session_state:
                del st.session_state['llm']
                    
            st.session_state['llm_model'] = llm_model
            st.session_state['llm'] = load_llm(llm_model=st.session_state['llm_model'],
                                                   host=ollama_host,
                                                   temperature=temperature)
            
            if st.session_state.get('extra_retriever', None) == None:
                st.session_state['extra_retriever'] = load_extra_retriever()
                if st.session_state.get('extra_retriever', None) != None:            
                    st.toast("Reloaded extra docs")

            st.session_state['search'] = get_hashi_search(llm=st.session_state['llm'],
                                                          extra_retriever=st.session_state['extra_retriever'])
            st.session_state['chat'] = get_hashi_chat(llm=st.session_state['llm'],
                                                      extra_retriever=st.session_state['extra_retriever'])
            st.toast(f"Loaded {st.session_state.get('llm_model')} with {temperature}")
            # st.rerun(scope="fragment")

    if st.session_state.get('llm', None) == None:
        st.error("No models loaded. Please select and load a model.")
    else:
        st.info(f"Current model {st.session_state.get('llm_model')} with {temperature}")


if __name__ == "__main__":
    main()
