#!/usr/bin/env python3

import logging
from typing import Optional

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.memory import ConversationSummaryMemory
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser)
from langchain_community.utilities.duckduckgo_search import (
    DuckDuckGoSearchAPIWrapper)
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import create_retriever_tool

import ai.hashi_prompts as hashi_prompts
from ai.common import get_retriever, load_llm


def get_hashi_agent(llm=None, callback_manager=None, extra_retriever: Optional[BaseRetriever] = None):
    if llm == None:
        logging.debug("Loading a new LLM")
        loaded_llm = load_llm(callback_manager=callback_manager)
    else:
        logging.debug("Using the provided LLM")
        loaded_llm = llm

    tools = []

    retriever = get_retriever(loaded_llm, use_filters=False, extra_retriever=extra_retriever)
    if retriever is not None:
        # tool names must be valid function names for tool calling
        rag_tool = create_retriever_tool(
            retriever=retriever,
            name="hashicorp_rag",
            description="HashiCorp RAG is your main source of knowledge and should be always checked first. Input should be a search query.")
        tools.append(rag_tool)
    else:
        logging.warning("No embeddings found, the agent will only use web tools")

    ddg_wrapper = DuckDuckGoSearchAPIWrapper(time="d", max_results=2)
    ddg_search = DuckDuckGoSearchResults(api_wrapper=ddg_wrapper,
                                         num_results=2,
                                         handle_tool_error=True)
    tools.append(ddg_search)

    # the agent executor runs synchronously, so it needs the sync browser
    sync_browser = create_sync_playwright_browser()
    playwright_toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
    tools.extend(playwright_toolkit.get_tools())

    memory = ConversationSummaryMemory(
        llm=loaded_llm, memory_key="chat_history", return_messages=True,
        input_key="input",
        output_key="output"
        )

    agent = create_tool_calling_agent(loaded_llm, tools, hashi_prompts.agent_prompt())
    agent_executor = AgentExecutor(agent=agent,
                                   tools=tools,
                                   memory=memory,
                                   max_iterations=4,
                                   handle_parsing_errors=True,
                                   verbose=True)

    return agent_executor, memory


def start_agent():
    agent_executor, memory = get_hashi_agent()

    while True:
        print("")
        print("-" * 50)
        print("-" * 50)

        query = input("User: ")
        if query == "exit":
            break
        if query == "":
            continue

        # the executor saves the question and answer to memory
        result = agent_executor.invoke({"input": query})

        print()
        print("-" * 50)
        print("-" * 50)
        print()
        print("User: ", query)
        print()
        print("AI: ", result["output"])
        print()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    start_agent()
