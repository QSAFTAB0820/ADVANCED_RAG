import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import create_retriever_tool
from dotenv import load_dotenv
from tools import create_support_ticket, fetch_order_status, schedule_call #custom tools
from retriever import get_hybrid_retriever #custom function
from langchain.agents import create_agent

load_dotenv()

def create_support_agent():
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0) #Creates the language model object using GPT-4o
    
    # Set up retriever tool
    kb_file = "knowledge_base.pdf" if os.path.exists("knowledge_base.pdf") else "knowledge_base.txt"
    retriever = get_hybrid_retriever(kb_file)
    retriever_tool = create_retriever_tool(
        retriever,
        "search_knowledge_base",
        "Search the customer support knowledge base to answer user questions about policies, shipping, payment, and warranties."
    )
    
    # Define Tools
    tools = [
        retriever_tool,
        create_support_ticket,
        fetch_order_status,
        schedule_call
    ]
    
    system_message = "You are a helpful and intelligent customer support assistant for Pranathi Software Services. You have access to tools to search the company knowledge base, create support tickets, and schedule calls. Always rely on the knowledge base when asked general questions. If the user gives multiple or conflicting intents, ask for clarification if needed, otherwise execute multiple tools if appropriate. When confirming ticket creation, always include the mock ticket ID."
    
    # Create Agent
    agent = create_agent(model=llm, tools=tools, system_prompt=system_message)
    
    return agent
