import ai.hashi_prompts as hashi_prompts
import streamlit as st
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from streamlit.runtime.scriptrunner import get_script_run_ctx

from ui.streamlit_shared import StreamHandler, display_result


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
            "00 - Summarise meeting notes",
            "01 - Convert text to Feature Request format",
            "02 - Simplify the text language",
            "03 - Improve the writing of the text",
            "04 - Extract action points",
            "05 - Summarize this text",
            "06 - Change the text to a friendly tone",
            "07 - Change the text to a confident tone",
            "08 - Change the text to a professional tone",
            "09 - Make the text half of its length",
            "10 - Make the text double its length",
            "11 - Brainstorm ideas on this subject"
        ]

        prompt_selected = st.selectbox("Select the action", options=prompt_options)
        if prompt_selected != None:
            if prompt_selected.startswith("00"):
                prompt_selected = ("Use Markdown in your answer. Use the following format for the input meeting notes \n",
                                   "### Agenda: \n",
                                   "High-level main topics that were discussed in the meeting notes. Only 5 at most.\n",
                                   "### Attendees: \n",
                                   "Names of people mentioned in the meeting notes \n",
                                   "### Notes: \n",
                                   "Rephrase and cleanup the meeting notes \n",
                                   "### Action points: \n",
                                   "Any actions points implied in the meeting notes\n",
                                   )

            elif prompt_selected.startswith("01"):
                prompt_selected = ("Use Markdown format in your answer. Convert the input text to the following Feature Request format \n",
                                   "### Summary/Description of Request (255 Characters or less): \n",
                                   "### Description & Workarounds (What use case or workflow is the customer trying to accomplish? What workarounds have they tried or other integrations are they using to accomplish this?): \n",
                                   "### Expected Outcome (What would the customer like to achieve from this request?): \n",
                                   "### Business Impact (How will this affect the customer's ability to use the product?): \n"
                                   )
            else:
                prompt_selected = prompt_selected[5:]
             
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
