from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import GigaChat
from langchain_ollama import OllamaEmbeddings
from urllib3.exceptions import InsecureRequestWarning
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
import warnings
import ssl

# Отключаем предупреждения SSL
warnings.filterwarnings("ignore", category=InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

CHROMA_PATH = "./db_metadata_v5"


def initialize_rag():
    print("Инициализация RAG-ассистента...")

    # Получаем ключ авторизации из переменных окружения
    authorization_key = os.getenv("GIGACHAT_AUTHORIZATION_KEY")
    if not authorization_key:
        raise ValueError("Не задана переменная GIGACHAT_AUTHORIZATION_KEY в .env файле")

    # Инициализируем LLM модель GigaChat с актуальным названием модели
    model = GigaChat(
        model="GigaChat",  # Используем актуальное название модели
        credentials=authorization_key,
        verify_ssl_certs=False,
        timeout=120,
        profanity_check=False,
        scope="GIGACHAT_API_PERS"
    )

    # Функция эмбеддинга
    embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

    # Загружаем Chroma
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Шаблон prompt'а для RAG
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            Вы - опытный юрист-консультант с именем 'Юридический Ассистент'. 
            Ваша задача — давать точные, профессиональные и структурированные ответы 
            на основе Конституции РФ и действующего законодательства.

            Основные правила:
            1. Отвечайте ТОЛЬКО на русском языке
            2. Базируйте ответы исключительно на предоставленном контексте
            3. Если информация отсутствует в контексте, отвечайте: 
               "Моя компетенция ограничена предоставленными юридическими документами. 
                Для детального ответа обратитесь к официальным источникам."
            4. Сохраняйте формально-деловой стиль общения
            5. Структурируйте ответы с использованием маркированных списков
            6. Цитируйте конкретные статьи законов из контекста
            7. Для сложных вопросов предлагайте пошаговый алгоритм действий
            8. Отказывайтесь отвечать на неправовые вопросы

            Контекст для ответа:
            {context}
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    # Создаём цепочку для комбинирования документов и LLM
    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt_template)

    return db, document_chain


def main():
    try:
        db, document_chain = initialize_rag()

        docs = db.get()
        num_docs = len(docs['documents']) if 'documents' in docs else 0
        print("✅ Инициализация прошла успешно!")
        print(f"🔎 Загружено документов из Chroma: {num_docs}")
        print("=" * 50)
        print("Добро пожаловать в RAG-ассистент!")
        print("Введите ваш вопрос или 'exit' для выхода")
        print("=" * 50)

        chat_history = []

        while True:
            user_input = input("Вы: ").strip()
            if user_input.lower() in ['exit', 'выход']:
                print("До свидания!")
                break
            if not user_input:
                continue

            results = db.similarity_search(user_input, k=3)
            documents = [
                Document(page_content=doc.page_content, metadata=doc.metadata)
                for doc in results
            ]

            inputs = {
                "question": user_input,
                "context": documents,
                "chat_history": chat_history
            }

            try:
                answer = document_chain.invoke(inputs)
                chat_history.append({"role": "human", "content": user_input})
                chat_history.append({"role": "assistant", "content": answer})

                print("\nАссистент:\n" + answer)
                print("-" * 50)
            except Exception as e:
                print(f"\nОшибка при обработке запроса: {str(e)}")
                print("-" * 50)
    except Exception as e:
        print(f"Ошибка при инициализации: {str(e)}")


if __name__ == "__main__":
    main()