import hashlib
import os
import shutil
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_PATH = "./db_metadata_v5"
DATA_PATH = "./knowledge_base"  # Папка с PDF файлами
global_unique_hashes = set()


def walk_through_files(path, file_extension='.pdf'):
    for dir_path, dir_names, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                yield os.path.join(dir_path, filename)


def load_documents():
    documents = []
    for file_path in walk_through_files(DATA_PATH, '.pdf'):
        try:
            loader = PyPDFLoader(file_path)
            pdf_pages = loader.load_and_split()
            documents.extend(pdf_pages)
        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
    return documents


def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        separators=[
            r'\nГлава\s+\d+',
            r'\nСтатья\s+\d+',
            r'\n\n',
            r'\n',
            ' '
        ]
    )

    chunks = []
    for doc in documents:
        cleaned = re.sub(r'\s+', ' ', doc.page_content)
        cleaned = re.sub(r'(Глава\s+\d+)', r'\n\1', cleaned)
        cleaned = re.sub(r'(Статья\s+\d+)', r'\n\1', cleaned)
        doc.page_content = cleaned.strip()

        doc_chunks = text_splitter.split_documents([doc])
        chunks.extend(doc_chunks)

    print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")

    unique_chunks = []
    for chunk in chunks:
        chunk_hash = hash_text(chunk.page_content)
        if chunk_hash not in global_unique_hashes:
            unique_chunks.append(chunk)
            global_unique_hashes.add(chunk_hash)

    print(f"[INFO] Unique chunks after deduplication: {len(unique_chunks)}")

    # 🔍 Вывод примеров чанков
    print("\n=== 🧩 Примеры загруженных чанков ===")
    for i, chunk in enumerate(unique_chunks[:5]):
        print(f"\n--- Chunk #{i + 1} ---")
        print(f"Source: {chunk.metadata.get('source', 'N/A')}")
        print(f"Length: {len(chunk.page_content)}")
        print(f"Content:\n{chunk.page_content[:700]}...\n")

    return unique_chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    try:
        db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model="mxbai-embed-large"),
            persist_directory=CHROMA_PATH
        )
        db.persist()
        print(f"[SUCCESS] Saved {len(chunks)} chunks to {CHROMA_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to save to Chroma: {e}")
        raise


def generate_data_store():
    documents = load_documents()
    if not documents:
        print("[WARN] No documents found!")
        return

    print(f"[INFO] Loaded {len(documents)} documents.")
    chunks = split_text(documents)
    save_to_chroma(chunks)


if __name__ == "__main__":
    generate_data_store()