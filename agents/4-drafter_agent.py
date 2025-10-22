from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

document_content=""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def update(content:str)->str:
    """
    Updates the document with the provided content
    """
    global document_content
    document_content = content
    return f"Document content successfully updated. The current document is : {document_content}"

@tool
def save(filename:str):
    """
    Saves the current document with the given filename and ends the process.
    Args:
        filename:str => Name for the text file
    """
    if not filename.endswith(".txt"):
        filename+=".txt"
    try:
        global document_content
        with open(filename, "w", encoding="utf-8") as file:
            file.write(document_content)
        print(f"Document saved with the filename: {filename}")
        return f"Document saved with the filename: {filename}"
    except Exception as e:
        return f"Error: {str(e)}"

drafting_tools = [update, save]
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(drafting_tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""
                You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
                
                - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
                - If the user wants to save and finish, you need to use the 'save' tool.
                - Make sure to always show the current document state after modifications.
                
                The current document content is:{document_content}
                """
    )
    if not state["messages"]:
        user_input = "Im ready to help you update my document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document?")
        print(f"user: {user_input}")
        user_message = HumanMessage(user_input)
    all_messages = [system_prompt] + list(state['messages']) + [user_message]
    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state:AgentState)->str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and # type: ignore
            "document" in message.content.lower()): # type: ignore
            return "end" # goes to the end edge which leads to the endpoint
        
    return "continue"


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(drafting_tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")


graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()


def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"): # type: ignore
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()

'''

 ===== DRAFTER =====

🤖 AI: [{'type': 'text', 'text': "I'm ready too! What content would you like to add to the document?", 'extras': {'signature': 'Cq4BAdHtim8KqMh+n2Cf5gR/Qt0DSXSZunkP7d5TDr7LC8odHNRdmX5DzSRzlLhiSPK9/kHe6rO50UUwV9CZjludStNN570+fI8v4sdyP9Psu1r0FNVIi6SAutnRRcnXB2Eao+w9tNayk5HVMNYa6MqUJCLkWnC4EDJv5aB5sSLJ7MwZKBTtrXfXYqAjq0Tx0SFESRnEgc4cMAplWkIEUZJayoHH57acE7z+gVJvDez1'}}]

What would you like to do with the document?write an email requesting for a holiday due to heavy rainfall
user: write an email requesting for a holiday due to heavy rainfall

🤖 AI: 
🔧 USING TOOLS: ['update']

 TOOL RESULT: Document content successfully updated. The current document is : Subject: Holiday Request - Heavy Rainfall

Dear [Manager's Name],

I am writing to request a holiday on [Date] due to the heavy rainfall and potential flooding in my area. The weather conditions are making it difficult and unsafe for me to commute to work.

I apologize for any inconvenience this may cause and will ensure all my urgent tasks are completed before my leave. I will be available remotely if needed.     

Thank you for your understanding.

Sincerely,
[Your Name]

What would you like to do with the document?address it to the college dean and add some college related issues. my name is utsav
user: address it to the college dean and add some college related issues. my name is utsav

🤖 AI:
🔧 USING TOOLS: ['update']

 TOOL RESULT: Document content successfully updated. The current document is : Subject: Holiday Request - Heavy Rainfall

Dear [Manager's Name],

I am writing to request a holiday on [Date] due to the heavy rainfall and potential flooding in my area. The weather conditions are making it difficult and unsafe for me to commute to work.

I apologize for any inconvenience this may cause and will ensure all my urgent tasks are completed before my leave. I will be available remotely if needed.     

Thank you for your understanding.

Sincerely,
[Your Name]

 TOOL RESULT: Document content successfully updated. The current document is : Subject: Holiday Request - Heavy Rainfall

Dear College Dean,

I am writing to request a holiday on [Date] due to the heavy rainfall and potential flooding in my area. The weather conditions are making it difficult and unsafe for me to commute to college, which could lead to missed classes and assignments.

I apologize for any inconvenience this may cause and will ensure all my urgent academic tasks are completed before my leave. I will be available remotely if needed for any urgent college matters.

Thank you for your understanding.

Sincerely,
Utsav

What would you like to do with the document?all good now
user: all good now

🤖 AI:
🔧 USING TOOLS: ['save']

 TOOL RESULT: Document content successfully updated. The current document is : Subject: Holiday Request - Heavy Rainfall

Dear College Dean,

I am writing to request a holiday on [Date] due to the heavy rainfall and potential flooding in my area. The weather conditions are making it difficult and unsafe for me to commute to college, which could lead to missed classes and assignments.

I apologize for any inconvenience this may cause and will ensure all my urgent academic tasks are completed before my leave. I will be available remotely if needed for any urgent college matters.

Thank you for your understanding.

Sincerely,
Utsav
Document saved with the filename: Holiday Request - Heavy Rainfall.txt

 TOOL RESULT: Document saved with the filename: Holiday Request - Heavy Rainfall.txt

 ===== DRAFTER FINISHED =====
'''