from typing import List

from langchain.agents import (AgentExecutor, AgentType, create_react_agent,
                              create_self_ask_with_search_agent,
                              create_structured_chat_agent, initialize_agent,
                              load_tools)
from langchain.agents.agent_toolkits.conversational_retrieval.tool import (
    create_retriever_tool)
from langchain.memory import ConversationSummaryMemory
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser)
from langchain_community.utilities import (SearxSearchWrapper,
                                           TextRequestsWrapper)
from langchain_community.utilities.duckduckgo_search import (
    DuckDuckGoSearchAPIWrapper)
from langchain_community.utils.ernie_functions import (
    convert_pydantic_to_ernie_function)
from langchain_core.runnables import (RunnableLambda, RunnableParallel,
                                      RunnablePassthrough)
from langchain_core.tools import Tool
from langchain_core.utils.function_calling import (
    convert_pydantic_to_openai_function, format_tool_to_openai_function)
from pydantic import BaseModel, Field

import ai.hashi_prompts as hashi_prompts
from ai.hashi_chat import get_retriever


def get_hashi_agent(callback_manager=None):


    tools = []

    # better tool descriptions. what are they for?
    # add tool to load URLs, so agent can check they are valid resources
    # add tool to load git repos, so agent can check they are valid resources
    # add tool to load web pages, so agent can check they are valid resources
    # move prompts into a separate file.
    
    rag_tool = create_retriever_tool(
        retriever=get_retriever(use_filters=False),
        name="HashiCorp RAG",
        description="HashiCorp RAG is your main source of knowledge and should be always checked first. Input should be a search query.")
    tools.append(rag_tool)

    ddg_wrapper = DuckDuckGoSearchAPIWrapper(time="d", max_results=2)
    ddg_search = DuckDuckGoSearchResults(api_wrapper=ddg_wrapper,
                                         num_results=2,
                                        #  return_direct=True,
                                         handle_tool_error=True)
    tools.append(ddg_search)
    
    # requests_tools = load_tools(["requests_all"])
    # tools.extend(requests_tools)
    from langchain_community.tools.playwright.current_page import (
        CurrentWebPageTool)
    
    async_browser = create_async_playwright_browser()
    playwright_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools.extend(playwright_toolkit.get_tools())
        
    # prompt = hub.pull("hwchase17/react")
    # prompt = hub.pull("langchain-ai/react-agent-template")
    # prompt = hub.pull("langchain-ai/react")
    # prompt = hub.pull("langchain-ai/react-2")
    # prompt = QA_CHAIN_PROMPT

    llm = load_llm(callback_manager)
    memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True
    )
    
    
    class Response(BaseModel):
        """Final response to the question being asked"""

        answer: str = Field(description="The final answer to respond to the user")
        sources: List[int] = Field(
            description="List of page chunks that contain answer to the question. Only include a page chunk if it contains relevant information"
        )
        
    llm_with_tools = llm.bind(
        # functions=[
        # # # The retriever tool
        # #     format_tool_to_openai_function(retriever_tool),
        #     # Response schema
        #     convert_pydantic_to_ernie_function(Response),
        # ],
        tools=tools,
        )
    
    # for tool in tools:
    #     tool.name
    agent = (
        {
            "question": lambda x: x["question"],
            "tools": {tool.__ne__: tool for tool in tools}, #itemgetter(tools), # RunnableLambda(lambda x: tools),
            "tool_names": {tool.__ne__: tool.name for tool in tools} , #RunnableLambda(lambda x: tools[0].name),
            # Format agent scratchpad from intermediate steps
            "agent_scratchpad": lambda x: x["intermediate_steps"],
        }
        | hashi_prompts.MISTRAL_INSTRUCT_AGENT_PROMPT
        | llm_with_tools
        | parse
    )


    # agent = create_react_agent(llm,
    #                            tools,
    #                         #    hub.pull("langchain-ai/react-agent-template"),
    #                            hashi_prompts.MISTRAL_INSTRUCT_AGENT_PROMPT,
    #                            )
    
    # agent = create_structured_chat_agent(llm,
    #                                      tools,
    #                                      hub.pull("hwchase17/structured-chat-agent")
    #                                      )

    # prompt = hub.pull("hwchase17/self-ask-with-search")
    # agent = create_self_ask_with_search_agent(llm, [rag_tool], prompt)
    
    # inputs = {"input": "What's a Vault client?"}
    # result = agent_executor.invoke(inputs)
    # print (result)

    # agent = initialize_agent(
    #     tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    #     verbose=True
    #     )
    # from langchain.chains import LLMMathChain
    agent_executor = AgentExecutor(agent=agent,
                                   tools=tools,
                                   memory=memory,
                                   max_iterations=4,
                                   handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax",
                                #    handle_parsing_errors=True,
                                #    early_stopping_method="generate",
                                #    return_intermediate_steps=True,
                                   verbose=True)

    return agent_executor, memory
   

import json

from langchain_core.agents import AgentActionMessageLog, AgentFinish


def parse(output):
    # If no function was invoked, return to user
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)

    # Parse out the function call
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])

    # If the Response function was invoked, return to the user with the function inputs
    if name == "Response":
        return AgentFinish(return_values=inputs, log=str(function_call))
    # Otherwise, return an agent action
    else:
        return AgentActionMessageLog(
            tool=name, tool_input=inputs, log="", message_log=[output]
        )


def start_agent():
    agent_executor, memory = get_hashi_agent()
    
    # while True:
    print("")
    print("-" * 50)
    print("-" * 50)
    
    # query = input("User: ")
    # if query == "exit":
    #     break
    # if query == "":
    #     continue
    
    # inputs = {"input": query,
    #           "instructions": hashi_prompts.INSTRUCTIONS 
    #         }
    question = "What's a Vault client?"
    inputs = {"question": question }
    result = agent_executor.invoke(inputs, return_only_outputs=True)


    try:
        memory.save_context(inputs, {"output": result["output"].content})

        print()
        print("-" * 50)
        print("-" * 50)
        print()
        print("User: ", inputs)
        print()
        print("AI: ", result["output"].content)
        print()
        print("Sources: ", )
        if "docs" in result:
            for d in result["docs"]:
                print (d.metadata.get('source', "source not found"))
            print(result)
        else:
            print("No sources found in ", result)
    except:
        print ("Error: ", result)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    start_agent()