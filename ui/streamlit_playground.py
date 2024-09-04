
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from streamlit.runtime.scriptrunner import get_script_run_ctx

from ui.streamlit_shared import StreamHandler, display_result


def add_playground():
    # st.header("Prompt playground")
    st.markdown(("Here are some links that may help you design prompts: \n" +
             " * https://www.promptingguide.ai/ \n" +
             " * https://prompt-helper.com/ \n"))
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
            
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system ),
                    ("user", input)
                ])
            chain = prompt | language_model | StrOutputParser()
            
            result = chain.invoke({}, config=RunnableConfig(callbacks=[stream_handler]))
            st.session_state['pg_result'] = result
            
        display_result(st.session_state['pg_result'], container)
        st.session_state["writing_disabled"] = False

    
