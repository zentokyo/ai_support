import requests
import uuid
import os
from dotenv import load_dotenv
import warnings

from urllib3.exceptions import InsecureRequestWarning

load_dotenv()


class GigaChatAuthError(Exception):
    """Кастомное исключение для ошибок аутентификации GigaChat"""
    pass


def get_gigachat_token() -> str:
    """
    Получает access token для работы с GigaChat API

    Returns:
        str: Access token

    Raises:
        GigaChatAuthError: Если не удалось получить токен
    """
    # Конфигурация из переменных окружения
    AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    AUTHORIZATION_KEY = os.getenv("GIGACHAT_AUTHORIZATION_KEY")

    if not AUTHORIZATION_KEY:
        raise GigaChatAuthError("Не задан GIGACHAT_AUTHORIZATION_KEY в .env файле")

    # Формируем заголовки запроса
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {AUTHORIZATION_KEY.strip()}",
    }

    # Тело запроса
    data = {"scope": "GIGACHAT_API_PERS"}

    try:
        # Отключаем предупреждения SSL для тестовой среды
        warnings.filterwarnings("ignore", category=InsecureRequestWarning)

        # Отправляем запрос на получение токена
        response = requests.post(
            AUTH_URL,
            headers=headers,
            data=data,
            verify=False,
            timeout=10
        )

        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            error_msg = f"Ошибка аутентификации. Код: {response.status_code}. Ответ: {response.text}"
            raise GigaChatAuthError(error_msg)

    except requests.exceptions.RequestException as e:
        raise GigaChatAuthError(f"Ошибка подключения к API: {str(e)}")


if __name__ == "__main__":
    try:
        token = get_gigachat_token()
        print("Токен успешно получен!")
        print(f"Access token: {token[:15]}...")
    except GigaChatAuthError as e:
        print(f"Ошибка: {str(e)}")