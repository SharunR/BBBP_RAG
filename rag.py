from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_vector_db():

    # Load knowledge base
    loader = TextLoader(
        "data/bbbp_text.txt",
        encoding="utf-8"
    )

    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector store
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Save locally
    vectorstore.save_local("faiss_index")

    print("FAISS index saved successfully!")


if __name__ == "__main__":
    build_vector_db()