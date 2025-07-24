# llm_agent.py
import os
import uuid
from typing import List, Literal

# Groq model
from langchain_groq import ChatGroq

# Memory & tools
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import get_buffer_string
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# Chroma + embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# === Set your Groq API key ===
os.environ["GROQ_API_KEY"] = "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"

# === Load existing Chroma memory store ===
vectorstore = Chroma(
    persist_directory="jbnsts_chroma",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# === Utility to identify user ===
def get_user_id(config: RunnableConfig) -> str:
    uid = config["configurable"].get("user_id")
    if not uid:
        raise ValueError("Missing user_id in config")
    return uid

# === Define memory tools ===
@tool
#save_recall_memory: Saves user-provided memory to the vector store.
def save_recall_memory(mem: str, config: RunnableConfig) -> str:
    """Save a memory string to long-term vector memory for the current user."""
    user_id = get_user_id(config)
    doc = Document(page_content=mem, metadata={"user_id": user_id}, id=str(uuid.uuid4()))
    vectorstore.add_documents([doc])
    return "Memory saved."


@tool
#search_recall_memories: Retrieves user-relevant memories from the vector store.
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for memories relevant to a user's query."""
    user_id = get_user_id(config)
    docs = retriever.invoke(query)
    memories = [d.page_content for d in docs if d.metadata.get("user_id") == user_id]
    return "\n".join(memories) if memories else "No relevant memories found."



tools = [save_recall_memory, search_recall_memories]

# === Define state ===
class State(MessagesState):
    recall_memories: List[str]

# === Prompt template ===
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant with long-term memory. Context:\n{recall_memories}"),
    ("placeholder", "{messages}")
])

# === LLM setup ===
llm = ChatGroq(model_name="llama3-8b-8192").bind_tools(tools)

# === Agent logic ===
def agent(state: State) -> State:
    recall_text = "\n".join(state["recall_memories"])
    messages = state["messages"]
    output = (prompt | llm).invoke({"messages": messages, "recall_memories": recall_text})
    return {"messages": [output]}

def load_memories(state: State, config: RunnableConfig) -> State:
    convo = get_buffer_string(state["messages"])
    recall = search_recall_memories.invoke(convo, config)
    return {"recall_memories": recall}

def route_tools(state: State) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END

# === Build the graph ===
builder = StateGraph(State)
builder.add_node("load_memories", load_memories)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges("agent", route_tools, ["tools", END])
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=MemorySaver())

# === Chat runner ===
def chat(user_id: str, thread_id: str):
    while True:
        message = input("User: ")
        if message.strip().lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break
        cfg = {"configurable": {"user_id": user_id, "thread_id": thread_id}}
        state = {"messages": [("user", message)]}
        for chunk in graph.stream(state, config=cfg):
            for node, updates in chunk.items():
                if "messages" in updates:
                    msg = updates["messages"][-1]
                    print(f"{msg.type.capitalize()}: {msg.content}")

# === Example use ===
if __name__ == "__main__":
    chat("u1", "main")
