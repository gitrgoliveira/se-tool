import streamlit as st
from langchain.schema.runnable.config import RunnableConfig
from streamlit.runtime.scriptrunner import get_script_run_ctx

from ui.streamlit_shared import StreamHandler, display_result


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
                import traceback
                st.error(f"{ex}")
                st.error(traceback.format_exc()) 
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

