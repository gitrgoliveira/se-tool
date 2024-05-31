#!/usr/bin/env python3

import logging
from operator import itemgetter

from langchain.memory import ConversationSummaryMemory
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, format_document
from langchain_core.retrievers import BaseRetriever

import ai.hashi_prompts as hashi_prompts
from ai.common import get_retriever, load_llm

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def retrieval_qa_chain(llm: Ollama, retriever: BaseRetriever| None, memory):
    from langchain.chains.conversational_retrieval.prompts import (
        CONDENSE_QUESTION_PROMPT)
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    
    base_template = hashi_prompts.prompt_from_model(llm.model).format(
        system=("All questions are in the context of HashiCorp products. Given the following conversation and a follow up question, rephrase ",
                "the follow up question to be a standalone question, in its original language, tone and with corrected grammar and spelling."),
        prompt=("Chat History: \n",
                "{chat_history}",
                "\nFollow Up Input: {question} \n",
                "Standalone question:")
    )
    # CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template=base_template)
    
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"),
    )
    
    # Calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm #get_hf_llm()
        | StrOutputParser(),
    }
    
    # Retrieve the relevant documents
    if retriever == None:
        retrieved_documents = {
            "docs": {},
            "question": lambda x: x["standalone_question"],
        }

        # Construct the inputs for the final prompt
        final_inputs = {
            "context": {},
            "question": itemgetter("question"),
            "chat_history": lambda x: get_buffer_string(x.get("chat_history", [])),
        }
        
    else:
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | retriever,
            "question": lambda x: x["standalone_question"],
        }
        
        # Construct the inputs for the final prompt
        final_inputs = {
            "context": lambda x: _combine_documents(x["docs"]),
            "question": itemgetter("question"),
            "chat_history": lambda x: get_buffer_string(x.get("chat_history", [])),
        }
    
    # The part that returns the answers
    
    answer = {
        "answer": final_inputs | hashi_prompts.QA_prompt(llm.model) | llm,
        "docs": itemgetter("docs"),
    }
    
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer
        
    
    return final_chain


def get_hashi_chat(llm=None, callback_manager=None, extra_retriever: BaseRetriever = None):
    if llm == None:
        logging.debug("Loading a new LLM")
        loaded_llm = load_llm(callback_manager=callback_manager)
    else:
        logging.debug("Using the provided LLM")
        loaded_llm=llm
        
    memory = ConversationSummaryMemory(
        llm=loaded_llm, memory_key="chat_history", return_messages=True
        )
    retriever = get_retriever(loaded_llm, use_filters=True, extra_retriever=extra_retriever)
    
    qa = retrieval_qa_chain(
        llm=loaded_llm, retriever=retriever, memory=memory
    )
    
    return qa, memory

def start_chain():
    qa, memory = get_hashi_chat()
    
    while True:
        print("")
        print("-" * 50)
        print("-" * 50)
        
        query = input("User: ")
        if query == "exit":
            break
        if query == "":
            continue
    
        inputs = {"question": query}
        result = qa.invoke(inputs)
        print(result)
        memory.save_context(inputs, {"answer": result["answer"].content})

        print()
        print("-" * 50)
        print("-" * 50)
        print()
        print("User: ", query)
        print()
        print("AI: ", result["answer"].content)
        print()
        print("Sources: ", )
        if "docs" in result:
            for d in result["docs"]:
                print (d.metadata.get('source', "source not found"))
            print(result)
        else:
            print("No sources found in ", result)



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    start_chain()
    # start_agent()