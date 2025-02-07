import os
from typing import Iterator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from dotenv import load_dotenv

load_dotenv()
qdrant_url = os.getenv("QDRANT_URL_LOCALHOST")
chunk_size = 1000        # Adjust chunk size as needed
chunk_overlap = 100      # Adjust overlap as needed

class DoclingTextLoader(BaseLoader):
    def __init__(self, file_path: str | list[str]) -> None:
        # Accepts either a single file path or a list of file paths.
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            # If the file is huge, consider reading in blocks or processing line-by-line.
            with open(source, "r", encoding="utf-8") as f:
                text = f.read()
            yield LCDocument(page_content=text)

def create_vector_database():
    # Specify the path to your large text file.
    loader = DoclingTextLoader(file_path=r"E:\chainlit\data\itr.txt")
    
    # Initialize the text splitter. You can experiment with different splitters 
    # (e.g., splitting by sentences or paragraphs) if the structure of your data allows.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    # Load documents from file(s). For huge files, ensure you have enough memory.
    docling_documents = list(loader.lazy_load())
    
    # Optionally, write out the full document(s) to an output file for debugging.
    with open('data/output_docling.md', 'a', encoding="utf-8") as f:
        for doc in docling_documents:
            f.write(doc.page_content + '\n')
    
    # Split the large document into smaller chunks.
    splits = text_splitter.split_documents(docling_documents)
    
    # Initialize embeddings (this step may take some time with many chunks).
    embeddings = FastEmbedEmbeddings()
    
    # Create and persist a Qdrant vector database from the chunked documents.
    # If the number of chunks is extremely high, consider processing them in batches.
    vectorstore = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        url=qdrant_url,
        collection_name="ITR-4",
    )
    
    print('Vector DB created successfully!')

if __name__ == "__main__":
    create_vector_database()
