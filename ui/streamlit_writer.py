import ai.hashi_prompts as hashi_prompts
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit_quill import st_quill

from ui.streamlit_shared import StreamHandler, display_result
from markdownify import markdownify as md


def hashi_writer ():
    col1, col2 = st.columns(2)
    
    with col1:
        TEXT_AREA_HEIGHT = 400
        text = ""
                    
        query = st_quill(value=text, placeholder="Paste your text here", html=True)
    

        prompt_options = [
            "00 - Summarise meeting notes",
            "01 - Convert text to Feature Request format",
            "02 - Simplify the text language",
            "03 - Improve how this text is written",
            "04 - Make the text more clear, concise and engaging. If possible, keep the same structure and tone.",
            "05 - Extract action points",
            "06 - Summarize this text",
            "07 - Generate a title",
            "08 - Create an email subject",
            "09 - Change the text to a friendly tone",
            "10 - Change the text to a confident tone",
            "11 - Change the text to a professional tone",
            "12 - Make the text half of its length",
            "13 - Make the text double its length",
            "14 - Generate an OKR from this text",
            "15 - Brainstorm ideas on this subject"
        ]

        prompt_selected = st.selectbox("Select the action", options=prompt_options)
        if prompt_selected != None:
            if prompt_selected.startswith("00"):
                prompt_selected = """
generate meeting notes, in Markdown format. The meeting notes must follow this format:
    ### Agenda:
        High-level main topics that were discussed in the meeting notes. Only 5 at most.
    ### Attendees:
        Names of people mentioned in the meeting notes
    ### Notes:
        Rephrase and cleanup the meeting notes
    ### Action points
     Any actions points implied in the meeting notes
"""

            elif prompt_selected.startswith("01"):
                prompt_selected = """
generate a customer's feature request description, with a comprehensive problem statement. Ensure the following elements are included:
    1. Detailed description of the problem the customer is experiencing.
    2. Desired outcome that the customer wants to achieve.
    3. Any workarounds or solutions that have already been tried.
    4. The business impact of the feature request.

Output:
- Problem Description: 
- Desired Outcome: 
- Attempted Workarounds: 
- Business Impact: 
"""
                                    
            elif prompt_selected.startswith("14"):
                prompt_selected = """
generate an OKR (Objectives and Key Results) based on the information provided. The OKR should include a clear, concise, and meaningful objective that describes what to accomplish within a specific timeframe. Additionally, provide 3-5 key results that are specific, measurable, achievable, relevant, and time-bound (SMART).


Example:
    Objective: Improve customer satisfaction with our support services.

    Key Results:

    * Achieve a customer satisfaction score of 90% or higher by the end of Q3.
    * Reduce average response time to customer inquiries to under 2 hours.
    * Implement a new customer feedback system by the end of Q2.
    * Train all support staff on the new system by the end of Q2.

Please ensure the OKR is aligned with the main goals and priorities mentioned in the text.
"""
            else:
                prompt_selected = prompt_selected[5:].lower()
             
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

                prompt = hashi_prompts.writer_prompt_template()
                chain = prompt | language_model | StrOutputParser()
                md_query = md(query)
                
                result = chain.invoke(
                    {
                        "instruction": prompt_selected,
                        "input": md_query
                    }, config=RunnableConfig(callbacks=[stream_handler]))
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
