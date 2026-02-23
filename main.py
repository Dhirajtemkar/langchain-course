import os
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from langsmith import traceable

tavily = TavilyClient()

@tool
def search_tool(query:str) -> str:
    """
    This is a search tool to search on the internet.
    args:
        query - string query to search on the internet
    result: string output of the search 
    """
    return tavily.search(query=query)

llm = ChatOllama(model="gpt-oss:20b", temperature=0)
tools = [search_tool]
search_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a search agent and you will use the search_tool to perform search on users query"
)

@traceable
def main():
    print("Hello from langchain-course!")
    response = search_agent.invoke({"messages":[HumanMessage(content="Can you let me know the weather in Mumbai.")]})
    print(response)

if __name__ == "__main__":
    main()
