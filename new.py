from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from typing import TypedDict, List
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda


# Load environment variables from .env file  
load_dotenv()

# Step 1: Initialize Groq LLM
llm = ChatGroq(
    model="llama3-8b-8192",  # ✅ Groq-supported model
    api_key=os.environ["GROQ_API_KEY"]
)

# Step 2: Initialize Chroma vector store for long-term memory
CHROMA_DB_DIR = "./long_term_memory_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

long_term_memory_store = Chroma(
    collection_name="long_term_memory",
    embedding_function=embedding_model,
    persist_directory=CHROMA_DB_DIR
)

# Step 3: Define the function for saving and retrieving long-term memories
def save_to_long_term_memory(text: str):
    """Save new memory to the Chroma vector store if not already present."""
    if text:
        # Check if memory already exists
        existing = long_term_memory_store.similarity_search(text, k=1)
        if existing and existing[0].page_content.strip() == text.strip():
            return
        
        # Add new memory to vector store
        document = Document(page_content=text)
        long_term_memory_store.add_documents([document])

        # No need for persist(), Chroma will automatically persist with the persist_directory

def retrieve_long_term_memories(query: str) -> List[str]:
    """Retrieve relevant long-term memories using a query."""
    docs = long_term_memory_store.similarity_search(query, k=3)  # Get top 3 related memories
    return [doc.page_content for doc in docs]

# Step 4: Define the LangGraph node function for memory handling
def groq_memory_fn(state):
    history = state.get("messages", [])
    user_input = state["input"]
    
    # Add user message to history (short-term memory)
    history.append(HumanMessage(content=user_input))
    
    # Retrieve long-term memories
    long_term_memories = retrieve_long_term_memories(user_input)
    
    # Combine short-term history with long-term memories for context
    full_context = "\n".join(long_term_memories) + "\n" + "\n".join([msg.content for msg in history])
    
    # Create a list of messages for the LLM
    messages_for_llm = [HumanMessage(content=full_context)]
    
    # Get LLM response based on the combined context
    response = llm.invoke(messages_for_llm)
    
    # Add LLM response to history (short-term memory)
    history.append(response)
    
    # Save the response or user input to long-term memory
    save_to_long_term_memory(user_input)
    save_to_long_term_memory(response.content)
    
    return {"messages": history, "input": user_input}

# Step 5: Build the LangGraph workflow
class GraphState(TypedDict):
    input: str
    messages: List

graph = StateGraph(GraphState)  # Pass the TypedDict, not a raw dict

graph.add_node("llm", RunnableLambda(groq_memory_fn))
graph.set_entry_point("llm")
graph.add_edge("llm", END)

# Step 6: Add memory checkpointing (short-term memory)
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Step 7: Use a consistent thread ID for memory
thread_id = "user-456"  # In a real application, use a unique ID per user or session

# Step 8: Interactive chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        break
    
    output = app.invoke({"input": user_input}, config={"configurable": {"thread_id": thread_id}})
    print("Bot:", output["messages"][-1].content)
