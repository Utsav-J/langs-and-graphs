import os
from typing import TypedDict, Annotated, Sequence, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.tools import tool
load_dotenv()

#setup
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

#load the pdf
# pdf_path = os.path.dirname(__file__).join("stock_data.pdf")
pdf_path = "./stock_data.pdf"
print(pdf_path)
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")
pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

#chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
pages_split = text_splitter.split_documents(pages)

#chroma
persist_directory = os.path.dirname(__file__)
collection_name = "stock_data"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vector_store = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

# retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":5}
)

#tool definition
@tool
def retriever_tool(query:str)->str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."
    results = []
    for i , doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    return "\n\n".join(results)

tools = [retriever_tool]
tools_dict = {our_tool.name: our_tool for our_tool in tools}
llm = llm.bind_tools(tools)

# graph state and router function
class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]

def should_continue(state:AgentState)->bool:
    result = state['messages'][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0 #type:ignore


# llm calling node definition
system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""
def call_llm(state:AgentState)->AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    result = llm.invoke(messages)
    return {"messages":[result]}

def take_action(state:AgentState)->AgentState:
    """Execute tool calls from the LLM's response."""
    tool_calls = state["messages"][-1].tool_calls #type:ignore
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

#making the actual graph finally

graph = StateGraph(AgentState)
graph.add_node("call_llm",call_llm)
graph.add_node("retriever_node", take_action)
graph.set_entry_point("call_llm")
graph.add_conditional_edges(
    "call_llm",
    should_continue,
    {
        True:"retriever_node",
        False:END
    }
)
graph.add_edge("retriever_node","call_llm")
rag_agent = graph.compile()

# runner function
def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


if __name__ == "__main__":
    running_agent()


'''
PDF has been loaded and has 9 pages
Created ChromaDB vector store!

=== RAG AGENT===

What is your question: how did snp 500 perfomr
Calling Tool: retriever_tool with query: S&P 500 performance
Result length: 4855
Tools Execution Complete. Back to the model!

=== ANSWER ===
The S&P 500 index had a strong year in 2024, delivering approximately a 25% total return (around +23% in price terms). This marks the second consecutive year of over 20% returns for the S&P 500, a feat not seen since the late 1990s. (Document 1)

A significant factor in the S&P 500's performance was the dominance of mega-cap technology stocks, particularly the "Magnificent 7" (Apple, Microsoft, Alphabet, Amazon, Meta, Nvidia, and Tesla). These companies collectively surged by an average of 64-67% in 2024 and accounted for over half (about 54%) of the S&P 500's total return for the year. (Document 3)

In contrast, smaller-cap stocks, as represented by the S&P 500 Equal-Weight index and the Russell 2000, had more modest gains of about 10-11% in 2024, highlighting that the rally was not evenly distributed across the market. (Document 1)

What is your question: exit

'''