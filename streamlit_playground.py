
import streamlit as st
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from streamlit.runtime.scriptrunner import get_script_run_ctx

import hashi_prompts
from streamlit_shared import StreamHandler, display_result


def add_playground():
    TEXT_AREA_HEIGHT = 250
    system = st.text_area(label="System", height=TEXT_AREA_HEIGHT, key="pg_system",
                        placeholder="System instruction", label_visibility="collapsed")

    input = st.text_area(label="Prompt", height=TEXT_AREA_HEIGHT, key="pg_input",
                        placeholder="Input for LLM", label_visibility="collapsed")

                 
    start_button = st.button("Process",
                             key="pg_start_button",
                            use_container_width=True,
                            type="primary",
                            disabled=st.session_state.get("writing_disabled", False))
    
    container=st.container(border=True)
    chat_box = container.empty()
    if not start_button:
        if 'pg_result' in st.session_state:
            display_result(st.session_state['pg_result'], container)
        st.session_state["writing_disabled"] = False
        return
    
    if start_button:
        st.session_state["writing_disabled"] = True
        with st.spinner("Processing..."):
            script_run_context = get_script_run_ctx(suppress_warning=True)
            stream_handler = StreamHandler(chat_box, display_method='write', ctx=script_run_context)
            language_model = st.session_state['llm']
            if language_model is None:
                st.error("No language model loaded. Please select a model from the sidebar.")
                return
            prompt_template = hashi_prompts.prompt_from_model(model_name=st.session_state['llm_model'])

            prompt = PromptTemplate.from_template(prompt_template.format(
            system=system, prompt=input))
            chain = prompt | language_model | StrOutputParser()
            
            result = chain.invoke({}, config=RunnableConfig(callbacks=[stream_handler]))
            st.session_state['pg_result'] = result
            
        display_result(st.session_state['pg_result'], container)
        st.session_state["writing_disabled"] = False

    
