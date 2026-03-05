from src.data_loader import load_all_documents
from src.embeddings import EmbeddingPipeline

if __name__ == "__main__":
    docs = load_all_documents("data")

    chunks = EmbeddingPipeline().chunk_documents(docs)
    chunkvectors = EmbeddingPipeline().embed_chunks(chunks)

    print(chunkvectors)
