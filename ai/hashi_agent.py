#!/usr/bin/env python3

import logging
from contextlib import contextmanager
from typing import Iterator, Optional

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.memory import ConversationSummaryMemory
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_community.utilities.duckduckgo_search import (
    DuckDuckGoSearchAPIWrapper)
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, StructuredTool, create_retriever_tool
from playwright.sync_api import sync_playwright

import ai.hashi_prompts as hashi_prompts
from ai.common import get_retriever, resolve_llm


def return_errors(tool: BaseTool) -> BaseTool:
    """Report tool failures to the model as observations, instead of ending the agent run."""
    def run(**kwargs):
        try:
            return tool.invoke(kwargs)
        except Exception as e:
            logging.warning(f"Tool {tool.name} failed: {e}")
            return f"{tool.name} failed: {e}"

    def invalid_arguments(e: Exception) -> str:
        return f"{tool.name} got invalid arguments: {e}"

    return StructuredTool.from_function(func=run,
                                        name=tool.name,
                                        description=tool.description,
                                        args_schema=tool.args_schema,
                                        handle_validation_error=invalid_arguments)


@contextmanager
def get_hashi_agent(llm=None, callback_manager=None, extra_retriever: Optional[BaseRetriever] = None) -> Iterator[AgentExecutor]:
    """Yield an agent executor, and close its browser on exit."""
    loaded_llm = resolve_llm(llm, callback_manager)

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

    ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=2)
    ddg_search = DuckDuckGoSearchResults(api_wrapper=ddg_wrapper, num_results=2)
    tools.append(ddg_search)

    # the agent executor runs synchronously, so it needs the sync browser
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            playwright_toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=browser)
            tools.extend(playwright_toolkit.get_tools())
            tools = [return_errors(tool) for tool in tools]

            memory = ConversationSummaryMemory(
                llm=loaded_llm, memory_key="chat_history", return_messages=True,
                input_key="input",
                output_key="output"
                )

            prompt = hashi_prompts.agent_prompt(has_rag=retriever is not None)
            agent = create_tool_calling_agent(loaded_llm, tools, prompt)
            yield AgentExecutor(agent=agent,
                                tools=tools,
                                memory=memory,
                                max_iterations=4,
                                handle_parsing_errors=True,
                                verbose=True)
        finally:
            browser.close()


def start_agent():
    with get_hashi_agent() as agent_executor:
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
            try:
                result = agent_executor.invoke({"input": query})
            except Exception as e:
                logging.error(f"The agent could not answer: {e}", exc_info=True)
                continue

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
