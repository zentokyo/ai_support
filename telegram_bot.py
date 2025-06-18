import os
import logging
from typing import Dict, List
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)
from dotenv import load_dotenv
import json
from pathlib import Path
from custom_gigachat import initialize_rag
from langchain_core.messages import HumanMessage, AIMessage

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HISTORY_DIR = Path("chat_histories")
HISTORY_DIR.mkdir(exist_ok=True)

# Состояния для ConversationHandler
MAIN_MENU, CHATTING = range(2)

# Инициализация RAG системы
db, document_chain = initialize_rag()


class TelegramChatWrapper:
    """Класс для управления историей диалогов пользователя"""

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.history_file = HISTORY_DIR / f"{user_id}.json"
        self.history: List[Dict] = []
        self.load_history()

    def load_history(self):
        """Загрузка истории из файла"""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.error(f"Ошибка загрузки истории: {e}")
                self.history = []
        else:
            self.history = []

    def save_history(self):
        """Сохранение истории в файл"""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения истории: {e}")

    def add_message(self, role: str, content: str):
        """Добавление сообщения в историю"""
        self.history.append({"role": role, "content": content})
        self.save_history()

    def clear_history(self):
        """Очистка истории диалога"""
        self.history = []
        if self.history_file.exists():
            try:
                self.history_file.unlink()
            except Exception as e:
                logger.error(f"Ошибка удаления файла истории: {e}")
        logger.info(f"История очищена для пользователя {self.user_id}")

    def get_langchain_messages(self):
        """Преобразование истории в формат LangChain"""
        messages = []
        for msg in self.history[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        return messages


# Клавиатура главного меню
main_menu_keyboard = [
    ["📢 Начать консультацию"],
    ["🧹 Завершить консультацию"],
    ["ℹ️ Помощь"]
]

main_menu_markup = ReplyKeyboardMarkup(
    keyboard=main_menu_keyboard,
    resize_keyboard=True,
    one_time_keyboard=False
)

# Клавиатура во время консультации
consultation_keyboard = [
    ["🧹 Завершить консультацию"],
    ["↩️ Вернуться в меню"]
]

consultation_markup = ReplyKeyboardMarkup(
    keyboard=consultation_keyboard,
    resize_keyboard=True,
    one_time_keyboard=False
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user = update.effective_user
    welcome_message = (
        f"👋 Приветствую, {user.first_name}!\n\n"
        "Я ваш персональный юридический ассистент.\n"
        "Могу помочь с вопросами по законодательству РФ:\n"
        "- Гражданский кодекс\n"
        "- Закон о защите прав потребителей\n"
        "- Воздушный кодекс\n"
        "- Обязательное страхование\n\n"
        "Выберите действие:"
    )
    await update.message.reply_text(welcome_message, reply_markup=main_menu_markup)
    return MAIN_MENU


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = (
        "📚 *Помощь по использованию бота*\n\n"
        "✅ *Начать консультацию* - задайте юридический вопрос\n"
        "🧹 *Завершить консультацию* - очистить историю диалога\n"
        "↩️ *Вернуться в меню* - прервать консультацию\n\n"
        "Пример вопроса:\n"
        "Как оформить наследство по завещанию?\n\n"
        "⚠️ *Важно:* Я отвечаю только на юридические вопросы!"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def begin_consultation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик начала консультации"""
    await update.message.reply_text(
        "💼 Вы в режиме консультации. Задайте ваш юридический вопрос.\n\n"
        "Я отвечу на основе:\n"
        "- Гражданского кодекса РФ\n"
        "- Закона о защите прав потребителей\n"
        "- Воздушного кодекса РФ\n"
        "- ФЗ об обязательном страховании\n\n"
        "Для очистки истории нажмите 'Завершить консультацию'",
        reply_markup=consultation_markup
    )
    return CHATTING


async def end_consultation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик завершения консультации"""
    user_id = update.effective_user.id
    chat_wrapper = TelegramChatWrapper(user_id)
    chat_wrapper.clear_history()

    await update.message.reply_text(
        "🧹 История диалога очищена. Вы можете начать новую консультацию.",
        reply_markup=main_menu_markup
    )
    return MAIN_MENU


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик сообщений пользователя в режиме консультации"""
    user_id = update.effective_user.id
    user_message = update.message.text

    if user_message == "↩️ Вернуться в меню":
        await update.message.reply_text(
            "Возвращаемся в главное меню.",
            reply_markup=main_menu_markup
        )
        return MAIN_MENU

    chat_wrapper = TelegramChatWrapper(user_id)
    chat_wrapper.add_message("user", user_message)

    try:
        # Показываем индикатор набора сообщения
        await update.message.reply_chat_action(action="typing")

        # Получаем историю в формате LangChain
        history = chat_wrapper.get_langchain_messages()

        # Поиск релевантных документов
        results = db.similarity_search(user_message, k=3)

        # Формируем входные данные для RAG
        inputs = {
            "question": user_message,
            "context": results,
            "chat_history": history
        }

        # Получаем ответ от RAG
        response = document_chain.invoke(inputs)

        # Сохраняем ответ в историю
        chat_wrapper.add_message("assistant", response)

        # Отправляем ответ пользователю
        await update.message.reply_text(response)

    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        await update.message.reply_text(
            "⚠️ Произошла ошибка при обработке вашего запроса. Попробуйте задать вопрос иначе."
        )

    return CHATTING


async def handle_unexpected_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик непредвиденных сообщений вне консультации"""
    response = (
        "ℹ️ Я вас не понял. Пожалуйста, используйте кнопки меню для навигации.\n\n"
        "Если вы хотите задать юридический вопрос, нажмите '📢 Начать консультацию'."
    )
    await update.message.reply_text(response, reply_markup=main_menu_markup)
    return MAIN_MENU


async def handle_media_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик медиа-файлов (фото, видео, документы)"""
    response = (
        "⚠️ Я работаю только с текстовыми сообщениями.\n\n"
        "Пожалуйста, задайте ваш вопрос текстом или используйте кнопки меню."
    )
    await update.message.reply_text(response, reply_markup=main_menu_markup)
    return MAIN_MENU


async def handle_consultation_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик медиа-файлов во время консультации"""
    response = (
        "⚠️ Я могу анализировать только текстовые юридические вопросы.\n\n"
        "Пожалуйста, опишите ваш вопрос текстом или используйте кнопки управления консультацией."
    )
    await update.message.reply_text(response, reply_markup=consultation_markup)
    return CHATTING


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отмена текущей операции"""
    await update.message.reply_text(
        "Действие отменено. Используйте меню для навигации.",
        reply_markup=main_menu_markup
    )
    return MAIN_MENU


def main():
    """Основная функция запуска бота"""
    # Создаем приложение
    application = Application.builder().token(TOKEN).build()

    # Настройка ConversationHandler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [
                MessageHandler(filters.Regex("^📢 Начать консультацию$"), begin_consultation),
                MessageHandler(filters.Regex("^ℹ️ Помощь$"), help_command),
                MessageHandler(filters.Regex("^🧹 Завершить консультацию$"), end_consultation),

                # Обработчик непредвиденных текстовых сообщений в главном меню
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_unexpected_input),

                # Обработчик медиа-сообщений в главном меню
                MessageHandler(filters.PHOTO | filters.VIDEO | filters.Document.ALL, handle_media_input),
            ],
            CHATTING: [
                MessageHandler(filters.Regex("^🧹 Завершить консультацию$"), end_consultation),
                MessageHandler(filters.Regex("^↩️ Вернуться в меню$"), cancel),

                # Обработчик текстовых сообщений в режиме консультации
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message),

                # Обработчик медиа-сообщений в режиме консультации
                MessageHandler(filters.PHOTO | filters.VIDEO | filters.Document.ALL, handle_consultation_media),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    # Регистрируем обработчики
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("start", start))

    # Запускаем бота
    logger.info("Бот запущен...")
    application.run_polling()


if __name__ == "__main__":
    main()