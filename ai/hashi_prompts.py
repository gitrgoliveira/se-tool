import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def QA_prompt() -> ChatPromptTemplate:
    messages = [
        (
            "system",
            (
            "You are a friendly assistant for question-answering tasks and an expert in HashiCorp technology. \n"
            "All questions are in the context of HashiCorp products. \n"
            "Use the chat history and the retrieved context to answer the question. \n"
            "If you don't know the answer, just say that you don't know and ask the user to clarify. Keep the answer concise. Use markdown format. Provide external references for verification. \n"
            )
        ),
        (
            "user",
            (
            "Chat History: {chat_history} \n" 
            "Question: {question} \n"
            "Context: {context} \n"
            "Answer:"
            )
        )
    ]
    
    return ChatPromptTemplate.from_messages(messages)


def search_prompt() -> ChatPromptTemplate:
    message = [
        (
            "system",
            (
                "You are a friendly search assistant and an expert in HashiCorp technology. \n"
                "All questions are in the context of HashiCorp products. \n"
                "Use only the pieces of retrieved context to answer the question. \n"
                "If you don't know the answer, just say you don't know. Keep the answer concise. Use markdown format. Provide external references for verification. \n"
            )
        ),
        (
            "user",
            (
                "Question: {question} \n"
                "Context: {context} \n"
                "Answer:"
            )
        )
    ]
    
    return ChatPromptTemplate.from_messages(message)



def agent_prompt(has_rag: bool = True) -> ChatPromptTemplate:
    if has_rag:
        tool_guidance = "Always check the hashicorp_rag tool first, and only use the web tools when it does not have the answer. \n"
    else:
        tool_guidance = "Use the web tools to find the answer. \n"

    return ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a friendly assistant for question-answering tasks and an expert in HashiCorp technology. \n"
                "All questions are in the context of HashiCorp products. \n"
                + tool_guidance +
                "If you don't know the answer, just say that you don't know. Keep the answer concise, in markdown format, and always add external references to your source of knowledge. \n"
            )
        ),
        MessagesPlaceholder("chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

def writer_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        (
            "system",
            ("You are an assistant that proposes an alternative way of writing, maintaining the same tone. "
            "Use the input text to {instruction}"),
        ),
        (
            "user",
            "Input text: {input}",
        ),
    ])
        