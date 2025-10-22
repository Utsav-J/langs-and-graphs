from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, List, Union
load_dotenv()

class AgentState(TypedDict):
    messages:List[Union[HumanMessage, AIMessage]]

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

def process(state:AgentState)->AgentState:
    res = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=res.content))
    print(f'AI: {res.content}')
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.set_entry_point('process')
graph.set_finish_point('process')
agent = graph.compile()

conversation_history=[]

user_input = input("you: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({
        'messages':conversation_history
    })
    print(result['messages'])
    conversation_history = result['messages']
    user_input = input("you: ")

'''
AI: Hi there! How can I help you today?
[HumanMessage(content='hi', additional_kwargs={}, response_metadata={}), AIMessage(content='Hi there! How can I help you today?', additional_kwargs={}, response_metadata={})]

you: hi im utsav. what is langchaina?                 
AI: Hi Utsav, nice to meet you! LangChain is a bla bla <ai slop>
[HumanMessage(content='hi', additional_kwargs={}, response_metadata={}), AIMessage(content='Hi there! How can I help you today?', additional_kwargs={}, response_metadata={}), HumanMessage(content='hi im utsav. what is langchaina?', additional_kwargs={}, response_metadata={}), AIMessage(content="Hi Utsav, nice to meet you!\n\nLangChain is a powerful **open-source framework** designed to help developers **build applications powered by large language models (LLMs)**.\n\nWhile LLMs are incredibly capable on their own, real-world applications often require them to do more than just generate text. They need to:\n*   Access external data (documents, databases, APIs).\n*   Remember past interactions (memory).\n*   Make decisions about which tools to use.\n*   Perform multi-step reasoning.\n\nThis is where LangChain comes in. It provides a structured way to **connect LLMs with other sources of data and computation**, allowing you to create more complex and intelligent applications.\n\nThink of it as a toolkit that makes it much easier to orchestrate and chain together different components to build sophisticated LLM-driven applications.\n\nHere are its core concepts and what you can do with it:\n\n### Key Concepts of LangChain:\n\n1.  **Chains:** These are sequences of calls to LLMs or other utilities. For example, you might have a chain that takes user input, formats it into a prompt, sends it to an LLM, and then processes the LLM's output.\n2.  **Agents:** Agents are more dynamic. They allow LLMs to make decisions about which tools to use (e.g., a search engine, a calculator, a database query tool) and in what order, based on user input. This enables multi-step reasoning and problem-solving.\n3.  **Memory:** LangChain provides ways to persist state between calls of a chain or agent. This is crucial for conversational applications where the LLM needs to remember past interactions and maintain context.\n4.  **Prompts:** It offers tools for constructing and managing prompts for LLMs, including prompt templates, few-shot examples, and output parsers.\n5.  **Integrations:** LangChain has robust integrations with a wide array of LLM providers (OpenAI, Hugging Face, Anthropic, etc.), data sources (databases, APIs, web scraping), and vector stores (for retrieval-augmented generation or RAG).\n\n### What you can build with LangChain:\n\n*   **Chatbots and conversational AI:** Build intelligent assistants that can maintain context over long conversations and integrate with external knowledge.\n*   **Question-Answering systems:** Create applications that can answer questions by querying specific documents, databases, or even the web.\n*   **Summarization tools:** Condense large texts from various sources.\n*   **Data extraction:** Pull specific information from unstructured text documents.\n*   **Code generation and analysis:** Assist with programming tasks, explain code, or generate tests.\n*   **Autonomous agents:** Systems that can plan and execute tasks using various tools.\n\nIn essence, LangChain simplifies the development of LLM-powered applications by providing a higher-level abstraction and a collection of ready-to-use components. It's available in both Python and JavaScript.\n\nDoes that give you a good initial understanding, Utsav? Or would you like to explore a specific aspect further?", additional_kwargs={}, response_metadata={})]

you: whats my name?
AI: Your name is Utsav. You told me that in your previous message! 😊
[HumanMessage(content='hi', additional_kwargs={}, response_metadata={}), AIMessage(content='Hi there! How can I help you today?', additional_kwargs={}, response_metadata={}), HumanMessage(content='hi im utsav. what is langchaina?', additional_kwargs={}, response_metadata={}), AIMessage(content="Hi Utsav, nice to meet you!\n\nLangChain is a powerful **open-source framework** designed to help developers **build applications powered by large language models (LLMs)**.\n\nWhile LLMs are incredibly capable on their own, real-world applications often require them to do more than just generate text. They need to:\n*   Access external data (documents, databases, APIs).\n*   Remember past interactions (memory).\n*   Make decisions about which tools to use.\n*   Perform multi-step reasoning.\n\nThis is where LangChain comes in. It provides a structured way to **connect LLMs with other sources of data and computation**, allowing you to create more complex and intelligent applications.\n\nThink of it as a toolkit that makes it much easier to orchestrate and chain together different components to build sophisticated LLM-driven applications.\n\nHere are its core concepts and what you can do with it:\n\n### Key Concepts of LangChain:\n\n1.  **Chains:** These are sequences of calls to LLMs or other utilities. For example, you might have a chain that takes user input, formats it into a prompt, sends it to an LLM, and then processes the LLM's output.\n2.  **Agents:** Agents are more dynamic. They allow LLMs to make decisions about which tools to use (e.g., a search engine, a calculator, a database query tool) and in what order, based on user input. This enables multi-step reasoning and problem-solving.\n3.  **Memory:** LangChain provides ways to persist state between calls of a chain or agent. This is crucial for conversational applications where the LLM needs to remember past interactions and maintain context.\n4.  **Prompts:** It offers tools for constructing and managing prompts for LLMs, including prompt templates, few-shot examples, and output parsers.\n5.  **Integrations:** LangChain has robust integrations with a wide array of LLM providers (OpenAI, Hugging Face, Anthropic, etc.), data sources (databases, APIs, web scraping), and vector stores (for retrieval-augmented generation or RAG).\n\n### What you can build with LangChain:\n\n*   **Chatbots and conversational AI:** Build intelligent assistants that can maintain context over long conversations and integrate with external knowledge.\n*   **Question-Answering systems:** Create applications that can answer questions by querying specific documents, databases, or even the web.\n*   **Summarization tools:** Condense large texts from various sources.\n*   **Data extraction:** Pull specific information from unstructured text documents.\n*   **Code generation and analysis:** Assist with programming tasks, explain code, or generate tests.\n*   **Autonomous agents:** Systems that can plan and execute tasks using various tools.\n\nIn essence, LangChain simplifies the development of LLM-powered applications by providing a higher-level abstraction and a collection of ready-to-use components. It's available in both Python and JavaScript.\n\nDoes that give you a good initial understanding, Utsav? Or would you like to explore a specific aspect further?", additional_kwargs={}, response_metadata={}), HumanMessage(content='whats my name?', additional_kwargs={}, response_metadata={}), AIMessage(content='Your name is Utsav. You told me that in your previous message! 😊', additional_kwargs={}, response_metadata={})]

you: exit
'''