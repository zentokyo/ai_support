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
    """Рекурсивный обход PDF-файлов"""
    for dir_path, dir_names, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                yield os.path.join(dir_path, filename)


def load_documents():
    """Загрузка PDF-документов"""
    documents = []
    for file_path in walk_through_files(DATA_PATH, '.pdf'):
        try:
            print(f"[INFO] Загрузка {file_path}")
            loader = PyPDFLoader(file_path)
            pdf_pages = loader.load()
            documents.extend(pdf_pages)
        except Exception as e:
            print(f"[ERROR] Не удалось загрузить {file_path}: {e}")
    return documents


def normalize_text(text):
    """Нормализация текста для дедупликации"""
    # Удаление лишних пробелов и специальных символов
    text = re.sub(r'\s+', ' ', text)
    # Приведение к нижнему регистру и удаление специальных символов
    return re.sub(r'[^\w]', '', text.lower())


def hash_text(text):
    """SHA-256 хэширование текста с нормализацией"""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode()).hexdigest()


def clean_legal_text(text):
    """Очистка юридического текста от ненужной информации"""
    # Удаление служебных комментариев
    text = re.sub(r'См\..*?(\n|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'Информация об изменениях:.*?(\n|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'в редакции.*?(\n|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'вступил[а-я]* в силу.*?(\n|$)', '', text, flags=re.DOTALL)

    # Удаление ссылок на законы
    text = re.sub(r'Федеральным законом от\s+[^.]*\s+в\s+(?:пункт|статью).*?(\n|$)', '', text, flags=re.DOTALL)

    # Очистка от временных отметок
    text = re.sub(r'\d{2}\.\d{2}\.\d{4}\s+Система ГАРАНТ', '', text)

    # Очистка от повторяющихся фраз
    text = re.sub(r'ГАРАНТ:\s*', '', text)

    return text.strip()


def split_text(documents: list[Document]):
    """Разделение текста на чанки с учетом юридической структуры"""
    # Настройки разделителя для юридических документов
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            # Структурные элементы с разными форматами
            r'\n\n(?:Статья|Глава|Пункт)\s+\d+[.,)]\s*',  # Статья 1., Статья 2)
            r'\n\n(?:Статья|Глава|Пункт)\s+[IVXLCDM]+',  # Римские цифры
            r'\n\n(?:Статья|Глава|Пункт)\s+\d+',  # Простые цифры
            # Параграфы и строки
            r'\n\n',
            r'\n',
            ' '
        ]
    )

    chunks = []
    for doc in documents:
        try:
            # Очистка текста
            cleaned_text = clean_legal_text(doc.page_content)

            # Добавление разрывов перед структурными элементами
            cleaned_text = re.sub(r'(?:Статья|Глава|Пункт)\s+[\dIVXLCDM]+',
                                  r'\n\n\g<0>', cleaned_text)

            # Создание нового документа с очищенным текстом
            new_doc = Document(
                page_content=cleaned_text,
                metadata=doc.metadata.copy()
            )

            # Разделение на чанки
            doc_chunks = text_splitter.split_documents([new_doc])

            current_chapter = None
            current_article = None

            # Обработка каждого чанка
            for chunk in doc_chunks:
                # Проверка на пустой чанк
                if not chunk.page_content.strip():
                    continue

                # Извлечение структурных элементов
                section_match = re.search(r'(Глава|Статья|Пункт)\s+([\dIVXLCDM]+)', chunk.page_content)

                if section_match:
                    section_type = section_match.group(1)
                    section_num = section_match.group(2)
                    section_full = f"{section_type} {section_num}"

                    # Обновление текущей главы или статьи
                    if section_type == "Глава":
                        current_chapter = section_full
                    elif section_type == "Статья":
                        current_article = section_full

                    # Добавление конкретного раздела в метаданные
                    chunk.metadata["law_section"] = section_full
                    chunk.metadata["section_type"] = section_type
                    chunk.metadata["section_number"] = section_num

                # Добавление текущей главы и статьи в метаданные
                if current_chapter:
                    chunk.metadata["chapter"] = current_chapter
                if current_article:
                    chunk.metadata["article"] = current_article

                chunks.append(chunk)

        except Exception as e:
            print(f"[ERROR] Ошибка при разделении документа: {e}")

    print(f"[INFO] Разделено {len(documents)} документов на {len(chunks)} чанков.")

    # Дедупликация чанков
    unique_chunks = []
    for chunk in chunks:
        chunk_hash = hash_text(chunk.page_content)
        if chunk_hash not in global_unique_hashes:
            unique_chunks.append(chunk)
            global_unique_hashes.add(chunk_hash)

    print(f"[INFO] Уникальных чанков после дедупликации: {len(unique_chunks)}")

    # Вывод примеров чанков
    print("\n=== Примеры загруженных чанков ===")
    for i, chunk in enumerate(unique_chunks):
        print(f"\n--- Чанк #{i + 1} ---")
        print(f"Источник: {chunk.metadata.get('source', 'N/A')}")
        print(f"Тип раздела: {chunk.metadata.get('section_type', 'N/A')}")
        print(f"Номер: {chunk.metadata.get('section_number', 'N/A')}")
        print(f"Глава: {chunk.metadata.get('chapter', 'N/A')}")
        print(f"Статья: {chunk.metadata.get('article', 'N/A')}")
        print(f"Длина: {len(chunk.page_content)}")
        print(f"Содержание:\n{chunk.page_content[:500]}...")

    return unique_chunks


def save_to_chroma(chunks: list[Document]):
    """Сохранение чанков в векторное хранилище"""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    os.makedirs(CHROMA_PATH)

    try:
        # Используем nomic-embed-text через Ollama
        embedding_function = OllamaEmbeddings(model="nomic-embed-text")

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=CHROMA_PATH,
            collection_metadata={"hnsw:space": "cosine"}  # Для cosine similarity
        )
        db.persist()
        print(f"[SUCCESS] Сохранено {len(chunks)} чанков в {CHROMA_PATH}")
    except Exception as e:
        print(f"[ERROR] Не удалось сохранить в Chroma: {e}")
        raise


def generate_data_store():
    """Основной процесс обработки данных"""
    documents = load_documents()
    if not documents:
        print("[WARN] Не найдено документов!")
        return

    print(f"[INFO] Загружено {len(documents)} страниц.")
    chunks = split_text(documents)
    save_to_chroma(chunks)


if __name__ == "__main__":
    generate_data_store()