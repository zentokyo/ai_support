from langchain_community.document_loaders import AsyncHtmlLoader
from bs4 import BeautifulSoup
import os
import re

FILE_TO_PARSE = "data/inputData/links.txt"
DIR_TO_STORE = "data/docs"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"


def get_links() -> list:
    try:
        with open(FILE_TO_PARSE, "r") as f:
            return [link.strip() for link in f.readlines()]
    except Exception as e:
        print(f"Error reading links: {e}")
        return []


async def async_loader(links):
    try:
        # Установка пользовательского агента
        os.environ["USER_AGENT"] = USER_AGENT

        # Загрузка HTML
        loader = AsyncHtmlLoader(links, verify_ssl=False)
        docs = loader.load()

        # Обработка каждой страницы
        for doc in docs:
            # Создаем объект BeautifulSoup
            soup = BeautifulSoup(doc.page_content, 'html.parser')

            # Удаляем ненужные элементы
            for selector in [
                'header',
                'footer',
                'nav',
                'script',
                'style',
                '.side-menu',
                '.breadcrumbs'
            ]:
                for element in soup.select(selector):
                    element.decompose()

            # Извлекаем основной контент
            main_content = soup.find('main') or soup.body

            # Очистка текста
            cleaned_text = main_content.get_text(separator='\n', strip=True)

            # Улучшенная обработка структуры
            cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
            cleaned_text = re.sub(
                r'(РАЗДЕЛ [IVXL]+|ГЛАВА \d+|Статья \d+)',
                r'\n## \1\n',
                cleaned_text
            )

            doc.page_content = cleaned_text

        # Сохранение результатов
        os.makedirs(DIR_TO_STORE, exist_ok=True)
        for idx, doc in enumerate(docs):
            filename = f"document_{idx}.md"
            with open(f"{DIR_TO_STORE}/{filename}", "w", encoding="utf-8") as f:
                f.write(doc.page_content)
            print(f"File {filename} saved successfully")

    except Exception as e:
        print(f"Processing error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio

    links = get_links()
    if links:
        asyncio.run(async_loader(links))
    else:
        print("No links found in input file")