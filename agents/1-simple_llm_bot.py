from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START,END
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class AgentState(TypedDict):
    messsages: List[HumanMessage]

def process(state:AgentState)->AgentState:
    response = llm.invoke(input=state['messsages'])
    print(f"AI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,'process')
graph.add_edge('process', END)
agent = graph.compile()

user_input = input("say something: ")
while user_input != "exit":
    res = agent.invoke(
        {
            'messsages':[HumanMessage(content=user_input)]
        }
    )
    # print(res)
    user_input = input("say something: ")
        
'''
output:

say something: hello
AI: Hello! How can I help you today?
{'messsages': [HumanMessage(content='hello', additional_kwargs={}, response_metadata={})]}
'''