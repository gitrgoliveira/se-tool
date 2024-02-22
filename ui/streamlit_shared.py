from typing import Any, Dict, List, Sequence
from uuid import UUID

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner import add_script_run_ctx


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown', ctx=None):
        add_script_run_ctx(ctx=ctx)
        self.container = container
        self.text = initial_text
        self.display_method = display_method
        self.ctx = ctx

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        add_script_run_ctx(ctx=self.ctx)
        self.text += token 
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")
        
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        add_script_run_ctx(ctx=self.ctx)
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function("\n")

        return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        add_script_run_ctx(ctx=self.ctx)
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function("\n")
        return super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        add_script_run_ctx(ctx=self.ctx)
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function("\n")

        return super().on_retriever_end(documents, run_id=run_id, parent_run_id=parent_run_id, **kwargs)


def display_result(result: str, container: DeltaGenerator):
    container._html("""
                <button onclick='copyTextToClipboard()'>Copy to clipboard</button>
                <div id='result' style='display: none;' >""" + result + """</div>
                <script>
                    function copyTextToClipboard() {
                        var textToCopy = document.getElementById('result').innerText;
                        navigator.clipboard.writeText(textToCopy).then(
                            function() { console.log("Copied to clipboard successfully!"); },
                            function(err) { console.error("Failed to copy text: ", err); }
                        );
                    }
                </script>
            """,height=30)
    container.markdown(result, unsafe_allow_html=True)


def get_model_list() -> List[str]:
    try:
        import ollama
        models = ollama.list()
    except:
        st.error("Error loading Ollama model list")
        return []
    return [model['name'] for model in models['models']]
  