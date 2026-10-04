import os
import logging
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class KnowledgeBaseUnavailableError(Exception):
    """Raised when the configured knowledge base cannot be loaded."""


def get_hybrid_retriever(file_path):
    if not os.path.isfile(file_path):
        logger.error("Knowledge base file does not exist: %s", file_path)
        raise KnowledgeBaseUnavailableError(
            "The support knowledge base is unavailable. Provide either "
            "'knowledge_base.pdf' or 'knowledge_base.txt' in the application "
            "directory, then restart the assistant."
        )

    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    try:
        docs = loader.load()
    except Exception as exc:
        logger.exception("Failed to load knowledge base file: %s", file_path)
        raise KnowledgeBaseUnavailableError(
            "The support knowledge base is unavailable. Check the configured "
            "knowledge base file and restart the assistant."
        ) from exc
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    bm25_retriever = BM25Retriever.from_documents(splits)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="db")
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5] 
    )
    #print(ensemble_retriever) # Debugging statement to check the retriever object
    return ensemble_retriever
