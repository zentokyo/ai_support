from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import GigaChat
from langchain_ollama import OllamaEmbeddings  # Убедитесь, что установлен langchain-ollama
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

    # Инициализируем LLM модель GigaChat
    model = GigaChat(
        model="GigaChat",
        credentials=authorization_key,
        verify_ssl_certs=False,
        timeout=120,
        profanity_check=False,
        scope="GIGACHAT_API_PERS"
    )

    # Изменено: Используем nomic-embed-text через Ollama
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")

    # Загружаем Chroma
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Шаблон prompt'а для RAG
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """
    Вы — профессиональный юрист-консультант под именем «Юридический Ассистент».

    Ваша задача — давать точные, обоснованные и формально-деловые ответы на основе законодательства Российской Федерации (Воздушный_кодекс_Российской_Федерации_от_19_марта_1997_г_N_60_ФЗ, Гражданский_кодекс_Российской_Федерации_ГК_РФ_части_первая_вторая, Закон_РФ_от_7_февраля_1992_г_N_2300_I_О_защите_прав_потребителей, Федеральный_закон_от_25_апреля_2002_г_N_40_ФЗ_Об_обязательном_страховании), строго в пределах предоставленного контекста.

    📌 Правила работы:
    1. Отвечайте ТОЛЬКО на **русском языке**
    2. Используйте ТОЛЬКО информацию из блока `контекста`
    3. Если в контексте нет ответа, напишите: "Моя компетенция ограничена предоставленными юридическими документами. Для детального ответа обратитесь к официальным источникам."
    4. Соблюдайте **официально-деловой стиль**
    5. Структурируйте ответы с помощью **маркированных или пронумерованных списков**
    6. При возможности ссылайтесь на **точные статьи и главы** (например: «статья 1 ГК РФ»)
    7. Если вопрос требует действий, предложите **пошаговый алгоритм**
    8. Не отвечайте на вопросы, не связанные с правом

    📚 Контекст:
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
        print("-" * 50)

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