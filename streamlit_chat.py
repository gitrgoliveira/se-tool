from typing import List

import ollama
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from streamlit.runtime.scriptrunner import get_script_run_ctx

import hashi_prompts
from hashi_chat import ModelDownloader, get_hashi_chat, load_llm, check_ollama_host
from streamlit_playground import add_playground
from streamlit_shared import StreamHandler, display_result

# llm_model = "solar:10.7b-instruct-v1-q6_K"
# llm_model = "solar:10.7b"
# llm_model = "mistral:7b-instruct-v0.2-q4_0"
# llm_model = "neural-chat:7b-v3.3-q4_0"


def hashi_assistant():
    st.header("Your personal HashiCorp Assistant")
    
    # with st.spinner("Loading language model..."):
    if 'qa' not in st.session_state:
        st.warning("Please load an LLM first")
        return
        
    qa, memory = st.session_state['qa']

    # Add a text input for the search query
    query = st.text_area("Enter your search query:", height=200)
    print("assistant query:", query)
    search_button = st.button("Search", use_container_width=True,
                              type="primary",
                              disabled=st.session_state.get("search_button_disabled", False))
    print("assistant button:", search_button)

    # If a query has been entered, search the documents and display the results
    if search_button:
        with st.spinner("Searching..."):
            try:
                st.session_state["search_button_disabled"] = True
                # try: 
                ctx = get_script_run_ctx(suppress_warning=True)
                chat_box = st.empty()
                stream_handler = StreamHandler(chat_box, display_method='write', ctx=ctx)    
                inputs = {"question": str(query).strip()}
                result = qa.invoke(inputs, config=RunnableConfig(callbacks=[stream_handler]))
                memory.save_context(inputs, {"answer": result["answer"]})
                st.session_state['assistant_result'] = result
            except Exception as ex:
                st.error(f"{ex}")
                st.error(f"LLM Model: {st.session_state.get('llm_model', None)}")
                st.error(f"LLM: {st.session_state.get('llm', None)}")
                st.error(f"QA: {st.session_state.get('qa', None)}")
            finally:
                st.session_state["search_button_disabled"] = False
    
    if 'assistant_result' in st.session_state:
        result = st.session_state['assistant_result']
        st.success("Search results found!")
        try:
            display_result(result["answer"], st.container(border=True))
            # st.markdown(result["answer"])
            # chat_box.empty()

            with st.sidebar:
                st.divider()
                if "docs" in result and \
                    len(result["docs"]) > 0:
                    st.sidebar.markdown("### Discovered Sources")
                    with st.spinner("Compiling sources..."):                    
                        for doc in result["docs"]:
                            with st.sidebar.container(border=True):
                                # st.sidebar.info(doc.state['query_similarity_score'])
                                st.sidebar.info(doc.metadata['relevance_score'])
                                st.sidebar.write(doc.page_content)
                                st.sidebar.json(doc.metadata)
                                
                else:
                    st.warning("No results found.")
                    st.json(result)
        except Exception as e:
            st.error(e)
            st.json(result)


def hashi_writer ():
    col1, col2 = st.columns(2)
    
    with col1:
        TEXT_AREA_HEIGHT = 400
        text = ""
        # if 'result' in st.session_state:
            # text = st.session_state['result']
            # st.session_state['input'] = st.session_state['result']
            # del st.session_state['result']
        
            # print ("reusing input: ",text)
            
        query = st.text_area(label="Text to process", height=TEXT_AREA_HEIGHT, key="input", value=text,
                            placeholder="Paste your text here", label_visibility="collapsed")

        prompt_options = [
            "0 - Summarise meeting notes",
            "1 - Convert text to Feature Request format",
            "2 - Simplify the input language",
            "3 - Improve the writing of the text",
            "4 - Extract action points",
            "5 - Change the input to a friendly tone",
            "6 - Change the input to a confident tone",
            "7 - Change the input to a professional tone",
            "8 - Make the input shorter",
            "9 - Make the input longer",
            "9 - Brainstorm ideas on this subject"
        ]

        prompt_selected = st.selectbox("Select the action", options=prompt_options)
        if prompt_selected != None:
            if prompt_selected [0] == "0":
                prompt_selected = ("Use Markdown format in your answer. Summarise the input meeting notes in the following format \n",
                                   "### Agenda: \n",
                                   "### Attendees: \n",
                                   "### Notes: \n",
                                   "### Action points from notes: \n",
                                   )

            elif prompt_selected[0] =="1":
                prompt_selected = ("Use Markdown format in your answer. Convert the input text to the following Feature Request format \n",
                                   "### Summary/Description of Request (255 Characters or less): \n",
                                   "### Description & Workarounds (What use case or workflow is the customer trying to accomplish? What workarounds have they tried or other integrations are they using to accomplish this?): \n",
                                   "### Expected Outcome (What would the customer like to achieve from this request?): \n",
                                   "### Business Impact (How will this affect the customer's ability to use the product?): \n"
                                   )
            else:
                prompt_selected = prompt_selected[3:]
             
        start_button = st.button("Process",
                                 use_container_width=True,
                                 type="primary",
                                 disabled=st.session_state.get("writing_disabled", False))
    
    with col2:
        container=st.container(border=True)
        chat_box = container.empty()
        # chat_box.markdown("#### Results will appear here")
        
        if not start_button:
            if 'write_result' in st.session_state:
                display_result(st.session_state['write_result'], container)
            st.session_state["writing_disabled"] = False
            return
        
        st.session_state["writing_disabled"] = True
        try:
            with st.spinner("Processing..."):
                script_run_context = get_script_run_ctx(suppress_warning=True)
                stream_handler = StreamHandler(chat_box, display_method='write', ctx=script_run_context)
                language_model = st.session_state.get('llm', None)
                if language_model is None:
                    st.error("No language model loaded. Please select a model from the sidebar.")
                    return

                prompt = ChatPromptTemplate.from_template(hashi_prompts.writer_prompt_template(st.session_state['llm_model']))
                chain = prompt | language_model | StrOutputParser()
                
                result = chain.invoke({"instruction": prompt_selected, "input": query}, config=RunnableConfig(callbacks=[stream_handler]))
                st.session_state['write_result'] = result
                
        except Exception as ex:
            st.error(f"{ex}")
            st.error(f"LLM Model: {st.session_state.get('llm_model', None)}")
            st.error(f"LLM: {st.session_state.get('llm', None)}")
            st.error(f"QA: {st.session_state.get('qa', None)}")
            
        finally:
            if st.session_state.get('write_result', False):
                display_result(st.session_state['write_result'], container)
            st.session_state["writing_disabled"] = False


def get_model_list(ollama_host: str | None) -> List[str]:
    try:
        if ollama_host != None:
            ollama_host = check_ollama_host(ollama_host=ollama_host)
        cli = ollama.Client(host=ollama_host)
        models = cli.list()
    except Exception as e:
        st.error("Error loading Ollama model list {e}".format(e=e))
        return []
    return [model['name'] for model in models['models']]
    
# Create the Streamlit app
def main():
    st.set_page_config(
        page_title="HashiCorp SE AI Tools",
        layout="wide")
    
    st.title("Tools to Search, Research, and Write")
    import os
    with st.sidebar:
        st.sidebar.title("Ollama Settings")
        default_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        ollama_host = st.text_input("OLLAMA_HOST", key="OLLAMA_HOST", value=default_host, placeholder="OLLAMA_HOST", label_visibility="collapsed")
        ollama_host = check_ollama_host (ollama_host=ollama_host)
        
        col1, col2 = st.sidebar.columns([3,1])
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
            st.sidebar.error("No models found. Please run `ollama pull` to download models from https://ollama.ai/library")
        
        model_list = get_model_list(ollama_host=ollama_host)
        index = 0
        if 'llm_model' in st.session_state and model_list != []:
            index = model_list.index(st.session_state['llm_model'])
            
        llm_model = st.selectbox("Select the language model",
                                     options=model_list,
                                     label_visibility="collapsed",
                                     index=index)

        temperature = st.sidebar.slider("Less or more creative?", min_value=0.0, max_value=1.0, value=0.5, step=.1)
        load_llm_button = st.button("Load model", use_container_width=True, type="primary")    
        
        if load_llm_button:
            with st.spinner("Loading language model..."):
                if 'qa' in st.session_state:
                    del st.session_state['qa']
                if 'llm' in st.session_state:
                    del st.session_state['llm']
                    
                st.session_state['llm_model'] = llm_model
                st.session_state['llm'] = load_llm(llm_model=st.session_state['llm_model'],
                                                   host=ollama_host,
                                                   temperature=temperature)
                st.session_state['qa'] = get_hashi_chat(llm=st.session_state['llm'] )
                st.toast(f"Loaded {st.session_state.get('llm_model')} with {temperature}")

        if st.session_state.get('llm', None) == None:
            st.sidebar.error("No models loaded. Please select and load a model.")
        else:
            st.sidebar.info(f"Current model {st.session_state.get('llm_model')} with {temperature}")    
            
    
    search, assistant, writer, playground = st.tabs(["Search", "Research", "Write", "AI Playground"])    
    with search:
        hashi_assistant()
    with assistant:
        st.write("Coming soon!")
    with writer:
        hashi_writer()
    with playground:
        add_playground()


if __name__ == "__main__":
    main()
