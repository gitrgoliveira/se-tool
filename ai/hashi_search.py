#!/usr/bin/env python3

import logging
from operator import itemgetter

from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, format_document
from langchain_core.retrievers import BaseRetriever

import ai.hashi_prompts as hashi_prompts
from ai.common import get_retriever, load_llm


def simple_search(retrievers, query):
    from langchain.retrievers import EnsembleRetriever
    n_retrievers = len(retrievers)
    weights = [1 / n_retrievers] * n_retrievers
    ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)
    return ensemble_retriever.get_relevant_documents(query)


def retrieval_search_chain(llm: Ollama, retriever):
    
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    
    base_template = hashi_prompts.prompt_from_model(llm.model).format(
        system=("All questions are in the context of HashiCorp products.Given the following question, rephrase it as a standalone question, in its original language \n"),
        prompt=("Question: {question} \n",
                "Standalone question:")
    )
    
    pass_through = RunnablePassthrough()
    
    
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template=base_template)
    standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"]
    }
    | CONDENSE_QUESTION_PROMPT
    | llm     
    | StrOutputParser(),
    }
    # | get_hf_llm()
    
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
        }
    
    # The part that returns the answers
    
    answer = {
        "answer": final_inputs | hashi_prompts.search_prompt(llm.model) | llm,
        "docs": itemgetter("docs"),
    }
    
    final_chain = pass_through| standalone_question | retrieved_documents | answer
    
    return final_chain

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def get_hashi_search(llm=None, callback_manager=None, extra_retriever: BaseRetriever = None):
    if llm == None:
        logging.debug("Loading a new LLM")
        loaded_llm = load_llm(callback_manager=callback_manager)
    else:
        loaded_llm=llm
        
    retriever = get_retriever(loaded_llm, use_filters=True, multi_query=True, extra_retriever=extra_retriever)
    
    search = retrieval_search_chain(
        llm=loaded_llm, retriever=retriever
    )
    
    return search
    

def start_chain():
    qa = get_hashi_search()
    
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
        print()
        print("-" * 50)
        print("-" * 50)
        print()
        print("User: ", query)
        print()
        print("AI: ", result["answer"])
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