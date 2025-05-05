import hashlib
import os
import shutil
import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_PATH = "./db_metadata_v5"
DATA_PATH = "./data/docs"
global_unique_hashes = set()


def walk_through_files(path, file_extension='.md'):
    for (dir_path, dir_names, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                yield os.path.join(dir_path, filename)


def load_documents():
    documents = []
    for f_name in walk_through_files(DATA_PATH):
        try:
            loader = TextLoader(f_name, encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {f_name}: {str(e)}")
    return documents


def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def split_text(documents: list[Document]):
    # Кастомный сплиттер для юридических документов
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        separators=[
            r'\n## Статья \d+',
            r'\n## Глава \d+',
            r'\n### Раздел \d+',
            r'\n#### Часть \d+',
            '\n\n',
            '\n',
            ' '
        ]
    )

    chunks = []
    for doc in documents:
        # Предварительная обработка текста
        cleaned_content = re.sub(r'\s+', ' ', doc.page_content)
        cleaned_content = re.sub(
            r'(Статья \d+)',
            r'\n## \1',
            cleaned_content
        )
        doc.page_content = cleaned_content

        # Разделение документа
        doc_chunks = text_splitter.split_documents([doc])
        chunks.extend(doc_chunks)

    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Удаление дубликатов
    unique_chunks = []
    for chunk in chunks:
        chunk_hash = hash_text(chunk.page_content)
        if chunk_hash not in global_unique_hashes:
            unique_chunks.append(chunk)
            global_unique_hashes.add(chunk_hash)

    print(f"Unique chunks after deduplication: {len(unique_chunks)}")
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
        print(f"Successfully saved {len(chunks)} chunks to {CHROMA_PATH}")
    except Exception as e:
        print(f"Error saving to Chroma: {str(e)}")
        raise


def generate_data_store():
    documents = load_documents()
    if not documents:
        print("No documents found!")
        return

    print(f"Loaded {len(documents)} source documents")
    chunks = split_text(documents)
    save_to_chroma(chunks)


if __name__ == "__main__":
    generate_data_store()