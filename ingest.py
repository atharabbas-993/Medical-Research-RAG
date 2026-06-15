import json
import os
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

load_dotenv()

def load_papers(path="data/papers.json"):
    # Load raw papers from disk
    with open(path, "r") as f:
        return json.load(f)

def convert_to_documents(papers):
    # Convert paper dicts to LangChain Documents
    docs = []
    for paper in papers:
        content = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        doc = Document(
            page_content=content,
            metadata={
                "title": paper["title"],
                "authors": paper["authors"],
                "year": paper["year"],
            }
        )
        docs.append(doc)
    return docs

def setup_llm():
    # Groq LLM free and fast
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )

def setup_embeddings():
    # Cohere cloud embeddings no local download
    return CohereEmbeddings(
        model="embed-english-light-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

def setup_vectorstore(embeddings):
    # Load existing ChromaDB from disk
    return Chroma(
        collection_name="medical_rag",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

def setup_hybrid_retriever(docs, vectorstore):
    # BM25 keyword search on raw documents
    bm25_retriever = BM25Retriever.from_documents(docs, k=3)

    # Dense semantic search from ChromaDB
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Combine both retrievers with equal weight
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )
    return hybrid_retriever

def setup_multi_query(hybrid_retriever, llm):
    # Generate multiple medical phrasings of one question
    return MultiQueryRetriever.from_llm(
        retriever=hybrid_retriever,
        llm=llm
    )

def format_docs(docs):
    # Format docs with title and year as citation
    formatted = []
    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        year = doc.metadata.get("year", "N/A")
        authors = doc.metadata.get("authors", "N/A")
        source = f"[{year}] {title} — {authors}"
        formatted.append(f"{source}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

def build_rag_chain(retriever, llm):
    # Medical RAG prompt with citation instruction
    prompt = ChatPromptTemplate.from_template("""
You are a medical research assistant helping doctors and researchers.
Answer the question using ONLY the research papers provided below.
Always mention the paper title and year when referencing findings.
If the answer is not in the papers say "No relevant research found in the database."

Research Papers:
{context}

Question: {question}

Answer (with paper references):
""")
    # Full RAG chain query to answer
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    print("Setting up Medical RAG pipeline...\n")

    # Load and convert papers
    papers = load_papers()
    docs = convert_to_documents(papers)

    # Setup core components
    llm = setup_llm()
    embeddings = setup_embeddings()
    vectorstore = setup_vectorstore(embeddings)

    # Build retriever pipeline
    hybrid_retriever = setup_hybrid_retriever(docs, vectorstore)
    print("✅ Hybrid search ready")

    multi_query_retriever = setup_multi_query(hybrid_retriever, llm)
    print("✅ Multi query retriever ready")

    # Build final RAG chain
    rag_chain = build_rag_chain(multi_query_retriever, llm)
    print("✅ RAG chain ready\n")

    # Test with real medical questions
    questions = [
        "What are the most effective treatments for Type 2 diabetes?",
        "What are the side effects of metformin in elderly patients?",
        "How does COVID-19 affect the cardiovascular system?",
    ]

    print("=" * 60)
    print("🏥 MEDICAL RESEARCH RAG — ANSWERS FROM PUBMED")
    print("=" * 60)

    for question in questions:
        print(f"\n❓ {question}")
        print("-" * 60)
        answer = rag_chain.invoke(question)
        print(f"💬 {answer}")
        print("=" * 60)