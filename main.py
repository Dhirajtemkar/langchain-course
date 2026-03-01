import os

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

load_dotenv()

# ___________init___________
embedding_model = os.environ["EMBEDDING_MODEL"]
embeddings = OllamaEmbeddings(model=embedding_model)

llm_model = os.environ["LLM_MODEL"]
llm = ChatOllama(model=llm_model, temperature=0)

index_name = os.environ["PINECONE_INDEX"]
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
context_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

chat_prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant. You will answer the question with the releavent context provided below:

{context}

question: {query}
""")


# _____uitls_____
def format_context(docs) -> str:
    "format the docs received from vector db and create a single string of context"
    return "\n\n".join(doc.page_content for doc in docs)


# RAW RAG LLM call without lcel implementation


def rag_llm_without_lcel(query: str) -> str:
    "perform retriving releavent data from vector store, adding it to the chat_prompt and generating llm response."

    # Retrieve context from vector store
    docs = context_retriever.invoke(query)

    context = format_context(docs)
    print("context received...")

    prompt = chat_prompt_template.format_messages(context=context, query=query)

    llm_response = llm.invoke(prompt)

    return llm_response.content


# RAG LLM implementation with LCEL
def rag_llm_with_lcel(query:str) -> str:
    " RAG retrieval and response generation using LCEL, better approach"
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("query") | context_retriever | format_context
        )
        | chat_prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain.invoke({"query": query})


if __name__ == "__main__":

    query = "what is Pinecone in machine learning?"

    print("=" * 60)
    print("DIRECT LLM RESPONSE: NO RAG")
    print("=" * 60)
    result = llm.invoke([HumanMessage(content=query)])
    print(result.content)

    print("=" * 60)
    print("RAG LLM RESPONSE: NO LCEL")
    print("=" * 60)
    print("Retriving relevent data from vector store...")
    result = rag_llm_without_lcel(query)
    print(f"LLM RESPONSE: {result}")

    print("=" * 60)
    print("RAG LLM RESPONSE: WITH LCEL")
    print("=" * 60)
    print("Retriving relevent data from vector store...")
    result = rag_llm_with_lcel(query)
    print(f"LLM RESPONSE: {result}")