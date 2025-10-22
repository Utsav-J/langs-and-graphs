from typing import TypedDict, List, Sequence, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int) -> int:
    """
    Adds two integers.
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """
    Subtracts two integers (a - b).
    """
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiplies two integers.
    """
    return a * b

current_tools=[add, subtract, multiply]
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(current_tools)

def model_call(state:AgentState)->AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability."
    )
    result = model.invoke([system_prompt] + state['messages']) # type: ignore
    return {"messages":[result]} # since we;re using the reducer function in annotated type, this is enough for automatic appending and updation


def should_continue(state:AgentState)->str:
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls: # type: ignore
        return "end"
    return "continue"

graph=StateGraph(AgentState)
graph.add_node("model",model_call)
graph.set_entry_point("model")

tool_node = ToolNode(tools=current_tools)
graph.add_node("tools",tool_node)

graph.add_conditional_edges(
    "model",
    should_continue,
    {
        "end":END,
        "continue":"tools"
    }
)
graph.add_edge("tools","model")
app = graph.compile()


def stream_message(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

input = AgentState(messages=[HumanMessage("add 3 and 4. multiply their result by 6. subtract 5 from the final result. then say a maths joke")])
stream_message(app.stream(input, stream_mode="values"))

'''
OUTPUT:

================================ Human Message =================================

add 3 and 4. multiply their result by 6. subtract 5 from the final result. then say a maths joke
================================== Ai Message ==================================
Tool Calls:
  add (0a3a8749-c1dc-4ec0-9eb2-6655bc541a5f)
 Call ID: 0a3a8749-c1dc-4ec0-9eb2-6655bc541a5f
  Args:
    a: 3
    b: 4
================================= Tool Message =================================
Name: add

7
================================== Ai Message ==================================
Tool Calls:
  multiply (4f78dbe2-95b1-4c4c-b431-2b909a9aedeb)
 Call ID: 4f78dbe2-95b1-4c4c-b431-2b909a9aedeb
  Args:
    a: 7
    b: 6
================================= Tool Message =================================
Name: multiply

42
================================== Ai Message ==================================
Tool Calls:
  subtract (ce8211f3-4d91-4867-8905-ac4d4251b393)
 Call ID: ce8211f3-4d91-4867-8905-ac4d4251b393
  Args:
    a: 42
    b: 5
================================= Tool Message =================================
Name: subtract

37
================================== Ai Message ==================================

The final result is 37.

Why was the equal sign so humble?
Because he knew he wasn't < or > anyone else!
'''