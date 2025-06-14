
from typing import Optional
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain

# Если у вас есть отдельный файл models.index
try:
    from models.index import ChatMessage
except ImportError:
    # Создаем класс ChatMessage, если он не импортировался
    from pydantic import BaseModel


    class ChatMessage(BaseModel):
        question: str

CHROMA_PATH = "../db_metadata_v5"


def initialize_rag():
    """Инициализация всех компонентов RAG"""
    print("Инициализация RAG-ассистента...")

    # Инициализация модели
    model = OllamaLLM(model="llama3.2:latest", temperature=0.1)

    # Функция для эмбеддингов
    embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

    # Подготовка базы данных
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    chat_history = {}  # история диалогов по session_id

    # Шаблон промпта
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                [INST]Вы - опытный юрист-консультант с именем 'Юридический Ассистент'. 
                Ваша задача - давать точные, профессиональные и структурированные ответы 
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

                Пример ответа:
                ## Согласно статье 15 Конституции РФ:
                - Конституция имеет высшую юридическую силу
                - Законы не должны противоречить Конституции
                - Общепризнанные принципы международного права являются частью правовой системы

                [/INST]

                [INST]Контекст для ответа:
                {context}[/INST]
            """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )

    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt_template)

    return db, chat_history, document_chain


def query_rag(db, chat_history, document_chain, message: ChatMessage, session_id: str = "") -> str:
    """
    Запрос к RAG системе
    """
    if session_id not in chat_history:
        chat_history[session_id] = []

    # Генерация ответа
    response_text = document_chain.invoke({
        "context": db.similarity_search(message.question, k=3),
        "question": message.question,
        "chat_history": chat_history[session_id]
    })

    # Обновление истории диалога
    chat_history[session_id].append(HumanMessage(content=message.question))
    chat_history[session_id].append(AIMessage(content=response_text))

    return response_text


def chat_interface():
    """Интерактивный интерфейс для общения с ассистентом"""
    db, chat_history, document_chain = initialize_rag()

    print("\n" + "=" * 50)
    print("Добро пожаловать в RAG-ассистент!")
    print("Введите ваш вопрос или 'exit' для выхода")
    print("=" * 50 + "\n")

    session_id = input("Введите ID сессии (или нажмите Enter для новой): ") or "default_session"

    while True:
        try:
            user_input = input("\nВы: ").strip()
            if user_input.lower() in ['exit', 'quit', 'выход']:
                break

            if not user_input:
                print("Пожалуйста, введите вопрос")
                continue

            message = ChatMessage(question=user_input)
            response = query_rag(db, chat_history, document_chain, message, session_id)

            print("\nАссистент:")
            print(response)
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nЗавершение работы...")
            break
        except Exception as e:
            print(f"\nПроизошла ошибка: {str(e)}")
            continue


if __name__ == "__main__":
    chat_interface()