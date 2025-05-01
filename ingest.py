import hashlib
import os
import shutil

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


CHROMA_PATH = "./db_metadata_v5"
DATA_PATH = "./data/docs"
global_unique_hashes = set()


def walk_through_files(path, file_extension='.txt'):
    for (dir_path, dir_names, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                yield os.path.join(dir_path, filename)


def load_documents():

    documents = []
    for f_name in walk_through_files(DATA_PATH):
        document_loader = TextLoader(f_name, encoding="utf-8")
        documents.extend(document_loader.load())

    return documents


def hash_text(text):
    hash_object = hashlib.sha256(text.encode())
    return hash_object.hexdigest()


def split_text(documents: list[Document]):

    text_splitter = MarkdownTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    unique_chunks = []
    for chunk in chunks:
        chunk_hash = hash_text(chunk.page_content)
        if chunk_hash not in global_unique_hashes:
            unique_chunks.append(chunk)
            global_unique_hashes.add(chunk_hash)

    print(f"Unique chunks equals {len(unique_chunks)}.")

    return unique_chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="mxbai-embed-large"),
        persist_directory=CHROMA_PATH
    )

    # Persist the database to disk
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


if __name__ == "__main__":
    generate_data_store()