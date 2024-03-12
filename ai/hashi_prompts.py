import logging

import ollama
from langchain_core.prompts import PromptTemplate


def prompt_from_model(model_name: str) -> str:
    template = ollama.show(model_name)['template']\
        .replace("{{ if .System }}", "")\
        .replace("{{ end }}", "")\
        .replace("{{- if .System }}", "")\
        .replace("{{- end }}", "")\
        .replace("{{ .Response }}<end_of_turn>", "")\
        .replace("{{ .System }}", " {system} ")\
        .replace("{{ .Prompt }}", " {prompt} ")
        
    if model_name.startswith("starling-lm"):
        template = ollama.show(model_name)['template'].replace("{{ .System }}", "{system}").replace("{{ .Prompt}}", "{prompt} ")

    logging.debug(f"Base template used: {template}")
    return template


# mistral_instruct_prompt_template = (
#     "<s> [INST] You are a friendly assistant for question-answering tasks and an expert in HashiCorp technology. \n"
#     "Use the following pieces of retrieved context to answer the question, together with the chat history and your own knowledge. \n"
#     "If you don't know the answer, just say that you don't know or ask questions to clarify. Keep the answer concise, in markdown format, and always add external references to your source of knowledge. [/INST] </s> \n"
    
#     "[INST]Chat History: {chat_history} \n" 
#     "Question: {question} \n"
#     "Context: {context} \n"
#     "Answer: [/INST]"
# )

def QA_prompt(model_name: str) -> PromptTemplate:
    template = prompt_from_model(model_name=model_name)
    prompt = template.format(
        system=(
            "You are a friendly assistant for question-answering tasks and an expert in HashiCorp technology. \n"
            "All questions are in the context of HashiCorp products. \n"
            "Use the following pieces of retrieved context to answer the question, together with the chat history and your own knowledge. \n"
            "If you don't know the answer, just say that you don't know and ask for clarification. Keep the answer concise. Use markdown format. Provide external references for verification. \n"
        ), prompt=(
            "Chat History: {chat_history} \n" 
            "Question: {question} \n"
            "Context: {context} \n"
            "Answer:"
        )
    )
    
    return PromptTemplate.from_template(prompt)
                                        # partial_variables={'chat_history': ''})

def search_prompt(model_name: str) -> PromptTemplate:
    template = prompt_from_model(model_name=model_name)
    prompt = template.format(
        system=(
            "You are a friendly search assistant and an expert in HashiCorp technology. \n"
            "All questions are in the context of HashiCorp products. \n"
            "Use only the pieces of retrieved context to answer the question. \n"
            "If you don't know the answer, just say you don't know. Keep the answer concise. Use markdown format. Provide external references for verification. \n"
        ), prompt=(
            "Question: {question} \n"
            "Context: {context} \n"
            "Answer:"
        )
    )
    
    return PromptTemplate.from_template(prompt)


# MISTRAL_INSTRUCT_PROMPT = PromptTemplate.from_template(mistral_instruct_prompt_template,
#                                                        partial_variables={'chat_history': ''})


mistral_instruct_agent_prompt_template = (
    "<s> [INST] You are a friendly assistant for question-answering tasks and an expert in HashiCorp technology. \n"
    "Always normalise and expand the Human question, before using any tool. \n"
    "If you don't know the answer, just say that you don't know. Keep the answer concise, in markdown format, and always add external references to your source of knowledge. \n"
    # "If you don't know the answer, just say that you don't know or ask questions to the Human to clarify. Keep the answer concise, in markdown format and always add references to your source of knowledge.  \n"
    "To answer the question, use the chat history, and at least one of the tools. Stop iterations if Thoughts are in invalid format and return the final answer. [/INST] </s>   \n"
    "[INST] \n\nTOOLS:\n------\n\nYou have access to the following tools:\n\n{tools}\n\nTo use a tool, please use the following format:\n\n```\nThought: Do I need to use a tool? Yes\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n```\n\nWhen you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:\n\n```\nThought: Do I need to use a tool? No\nFinal Answer: [your response here]\n```\n\nBegin!\n\nPrevious conversation history:\n{chat_history}\n\nNew input: {question}\n{agent_scratchpad} [/INST]"

    # "You have access to the following tools:\n\n{tools}\n\n"
    # "To use a tool, please use the following format: \n"
    # "```\n"
    # "Thought: Do I need to use a tool? Yes \n"
    # "Action: the action to take, should be one of [{tool_names}] \n"
    # "Action Input: the input to the action \n"
    # "Observation: the result of the action \n```"
    # "When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format: \n"
    # "```\nThought: Do I need to use a tool? No \n"
    # "Final Answer: [your response here]\n``` \n"
    # # "[INST]Chat History: {chat_history} \n" 
    # "Chat History: {chat_history} \n" 
    # "Question: {question} \n"
    # # "Context: {context} \n"
    # # "Thought: {agent_scratchpad}\n"
    # "Thought: {agent_scratchpad} [/INST] \n"
)
MISTRAL_INSTRUCT_AGENT_PROMPT = PromptTemplate.from_template(mistral_instruct_agent_prompt_template,
                                               partial_variables={
                                                   'chat_history': '',})

# INSTRUCTIONS = (
#     "<s> [INST] You are a friendly assistant for question-answering tasks and an expert in HashiCorp technology. \n"
#     "If you don't know the answer, just say that you don't know. Keep the answer concise, in markdown format and always add references to your source of knowledge. [/INST] </s> \n"
# )
# input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools']
# template='Answer the following questions as best you can. You have access to the following tools:\n\n{tools}\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}'

#    hub.pull("langchain-ai/react-agent-template"),
# input_variables=['agent_scratchpad', 'input', 'instructions', 'tool_names', 'tools']
# partial_variables={'chat_history': ''}
# template='{instructions}\n\nTOOLS:\n------\n\nYou have access to the following tools:\n\n{tools}\n\nTo use a tool, please use the following format:\n\n```\nThought: Do I need to use a tool? Yes\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the action\nObservation: the result of the action\n```\n\nWhen you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:\n\n```\nThought: Do I need to use a tool? No\nFinal Answer: [your response here]\n```\n\nBegin!\n\nPrevious conversation history:\n{chat_history}\n\nNew input: {input}\n{agent_scratchpad}'

def writer_prompt_template(model_name: str):
    
    template = prompt_from_model(model_name=model_name)
    # template = ollama.show(model_name)['template'].replace("{{ .System }}", "{system}").replace("{{ .Prompt }}", "{prompt}")
    
    # if model_name.startswith("starling-lm"):
    #     template = ollama.show(model_name)['template'].replace("{{ .System }}", "{system}").replace("{{ .Prompt}}", "{prompt} ")

    return template.format(
        system=("You are an assistant that proposes an alternative way of writing, using the same tone. ",
        "Use the input text to {instruction}"), prompt="input text: {input}")
    