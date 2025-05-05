import re
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


def clean_think_tags(text: str) -> str:
    """Удаляет теги <think> и их содержимое из текста"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def initialize_rag():
    """Инициализация всех компонентов RAG"""
    print("Инициализация RAG-ассистента...")

    # Инициализация модели DeepSeek R1 через Ollama
    model = OllamaLLM(
        model="deepseek-r1:latest",
        temperature=0.3,  # Небольшая креативность для связного текста
        top_k=40,  # Более широкий выбор токенов
        top_p=0.9,  # Баланс между точностью и разнообразием
        repeat_penalty=1.1,
        num_ctx=4096,  # Увеличенный контекст
        stop = ["English:", "Translation:", "англ."]
    )

    # Функция для эмбеддингов
    embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

    # Подготовка базы данных
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    chat_history = {}  # история диалогов по session_id

    # Шаблон промпта для DeepSeek
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                [INST]<<SYS>>Вы - официальный эксперт Конституции РФ. Строгие требования:
        
        1. ЯЗЫК:
           - Только русский язык в официальной редакции
           - Запрещены любые иностранные слова и фразы
           - При необходимости перевода терминов: "[термин] (в пер. с англ. - ...)"
        
        2. СТИЛЬ:
           - Официально-деловой стиль
           - Только утверждённые формулировки
           - Без интерпретаций и дополнений
        
        3. ФОРМАТ:
           ## Статья XX
           [Дословный текст статьи]
           (без разделов "Описание", "Применение")
        
        Пример корректного ответа:
        ## Статья 91
        Президент Российской Федерации обладает неприкосновенностью.
        
        Нарушения:
        - Использование английского (CRITICAL ERROR)
        - Добавление неофициальных пояснений
        <SYS>>[/INST]

                [INST]Контекст для анализа:
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

    # Очистка ответа от тегов <think>
    cleaned_response = clean_think_tags(response_text)

    # Обновление истории диалога
    chat_history[session_id].append(HumanMessage(content=message.question))
    chat_history[session_id].append(AIMessage(content=cleaned_response))

    return cleaned_response


def chat_interface():
    """Интерактивный интерфейс для общения с ассистентом"""
    db, chat_history, document_chain = initialize_rag()

    print("\n" + "=" * 50)
    print("Добро пожаловать в RAG-ассистент по Конституции РФ!")
    print("Используется модель: deepseek-r1:latest")
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
