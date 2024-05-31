import logging

import streamlit as st
from langchain_core.runnables import RunnableConfig
from streamlit.runtime.scriptrunner import get_script_run_ctx

from ui.streamlit_shared import StreamHandler, display_result


def hashi_chat():
    st.header("Your personal assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if 'chat' not in st.session_state:
        st.warning("Please load an LLM first")
        return
        
    chat, memory = st.session_state['chat']
    
    # Display chat messages from history on app rerun
    message_container = st.container(height=400)
    for message in st.session_state.messages:
        with message_container.chat_message(message["role"]):
            st.write(message["content"])


    if prompt := st.chat_input("Say something"):
        with message_container.chat_message("User"):
            st.write(prompt)
            st.session_state.messages.append({"role": "User", "content": prompt})
            
        with message_container.chat_message("assistant"):
            try:
                ctx = get_script_run_ctx(suppress_warning=True)
                chat_box = st.empty()
                stream_handler = StreamHandler(chat_box, display_method='write', ctx=ctx)    

                inputs = {"question": str(prompt).strip()}
                with st.spinner():
                    response = chat.invoke(inputs, config=RunnableConfig(callbacks=[stream_handler]))
                # response = st.write_stream(chat.stream(inputs, config=RunnableConfig(callbacks=[stream_handler])))
                st.session_state['chat_response'] = response
                final_response = response['answer']
                
                if "docs" in response and \
                    len(response["docs"]) > 0:
                    final_response += (f"\n#### Source documents")
                    for doc in response["docs"]:
                        if doc.metadata.get('url', False):
                            final_response += (f"\n * {doc.metadata['url']}")
                        elif doc.metadata.get('source', False):
                            final_response += (f"\n * {doc.metadata['source']}")

                else:
                    final_response += ("\n### Warning")
                    final_response += ("\n*No source documents found. The information provided by the AI is probably incorrect!*")
                    st.warning("*No source documents found. The information provided by the AI is probably incorrect!*")

                memory.save_context(inputs, {"answer":final_response})
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                st.write(final_response)
                
            except Exception as ex:
                if 'chat_response' in st.session_state:
                    del st.session_state['chat_response']
                st.error(ex)
                logging.error(ex,exc_info=True)
        
    if 'chat_response' in st.session_state:
        chat_response = st.session_state['chat_response']
        with st.sidebar:
            st.divider()
            if "docs" in chat_response and \
                len(chat_response["docs"]) > 0:
                st.sidebar.markdown("### Search details")
                with st.spinner("Compiling sources..."):                    
                    for doc in chat_response["docs"]:
                        with st.sidebar.container(border=True):
                            # st.sidebar.info(doc.state['query_similarity_score'])
                            st.sidebar.info(doc.metadata['relevance_score'])
                            st.sidebar.write(doc.page_content)
                            if doc.metadata.get('text_as_html', False):
                                del doc.metadata['text_as_html']
                            st.sidebar.json(doc.metadata)
                            
            else:
                st.warning("*No source documents found. The information provided by the AI is probably incorrect!*")
                st.json(chat_response)
