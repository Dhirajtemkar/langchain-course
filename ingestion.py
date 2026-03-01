import os
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter 
from dotenv import load_dotenv


load_dotenv()

embedding_model = "nomic-embed-text"
pinecone_index = "langchain-course-01"

if __name__ == '__main__':
    
    # Loading the document as langchain Document Object
    loader = TextLoader("./mediumblog1.txt", encoding='utf8')
    document = loader.load()

    #Chunking
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )

    chunks = text_splitter.split_documents(documents=document)

    #Initialize Embedding model for Embedding chunks
    embedding = OllamaEmbeddings(model=embedding_model)

    # Send embedded chunks to Pinecone Vector Db index
    print("Ingesting...")
    PineconeVectorStore.from_documents(
        chunks,
        embedding=embedding,
        index_name=pinecone_index
    )

