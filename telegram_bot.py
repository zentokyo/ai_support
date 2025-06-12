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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
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
rag_chat_history: Dict[str, List[Dict]] = {}


class TelegramChatWrapper:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.history_file = HISTORY_DIR / f"{user_id}.json"
        self.history: List[Dict] = []
        self.load_history()

    def load_history(self):
        if self.history_file.exists():
            with open(self.history_file, "r", encoding="utf-8") as f:
                self.history = json.load(f)
        else:
            self.history = []

    def save_history(self):
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self.save_history()

    def clear_history(self):
        self.history = []
        rag_chat_history[str(self.user_id)] = []
        self.save_history()

    def get_langchain_messages(self):
        messages = []
        for msg in self.history[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def get_recent_messages(self, count=5) -> str:
        if not self.history:
            return "История сообщений пуста."
        recent = self.history[-count * 2:]
        return "\n\n".join(
            f"👤 Вы: {recent[i]['content']}\n🤖 Бот: {recent[i + 1]['content']}"
            for i in range(0, len(recent) - 1, 2)
        )


main_menu_keyboard = [
    ["📢 Начать консультацию", "🧹 Очистить чат"],
    ["📄 История сообщений", "ℹ️ Помощь"]
]

main_menu_markup = ReplyKeyboardMarkup(
    keyboard=main_menu_keyboard,
    resize_keyboard=True,
    one_time_keyboard=False
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    welcome_message = (
        f"Привет, {user.first_name}! 👋\n\n"
        "Я — ваш персональный юридический ассистент. Я помогу вам с вопросами "
        "по Конституции РФ и законодательству.\n\n"
        "Выберите действие из меню ниже:"
    )
    await update.message.reply_text(welcome_message, reply_markup=main_menu_markup)
    return MAIN_MENU


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📌 *Помощь*\n\n"
        "Вы можете воспользоваться следующими функциями:\n"
        "📢 Начать консультацию — задать вопрос по законодательству\n"
        "🧹 Очистить чат — удалить историю общения\n"
        "📄 История сообщений — показать последние вопросы и ответы\n"
        "ℹ️ Помощь — показать это сообщение"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def clear_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_history = TelegramChatWrapper(user_id)
    chat_history.clear_history()
    await update.message.reply_text("История диалога очищена.")


async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_id = update.effective_user.id
    chat_wrapper = TelegramChatWrapper(user_id)

    if text == "📢 Начать консультацию":
        await update.message.reply_text(
            "Задайте ваш юридический вопрос.\n\n"
            "Примеры:\n• Какие права гарантирует статья 15 Конституции РФ?\n"
            "• Как регулируется право на образование в РФ?"
        )
        return CHATTING

    elif text == "🧹 Очистить чат":
        chat_wrapper.clear_history()
        await update.message.reply_text("История диалога очищена.")
        return MAIN_MENU

    elif text == "📄 История сообщений":
        recent = chat_wrapper.get_recent_messages()
        await update.message.reply_text(recent)
        return MAIN_MENU

    elif text == "ℹ️ Помощь":
        await help_command(update, context)
        return MAIN_MENU

    else:
        await update.message.reply_text("Пожалуйста, выберите действие из меню.")
        return MAIN_MENU


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text

    chat_wrapper = TelegramChatWrapper(user_id)
    chat_wrapper.add_message("user", user_message)

    try:
        # Вызов RAG с документацией
        rag_chat_history.setdefault(str(user_id), [])
        history = chat_wrapper.get_langchain_messages()

        results = db.similarity_search(user_message, k=3)
        inputs = {
            "question": user_message,
            "context": results,
            "chat_history": history
        }

        response = document_chain.invoke(inputs)

        chat_wrapper.add_message("assistant", response)
        await update.message.reply_text(response)

    except Exception as e:
        logger.error(f"Ошибка в RAG: {e}")
        await update.message.reply_text(
            "Произошла ошибка при обработке запроса. Попробуйте снова позже."
        )

    return CHATTING


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Консультация завершена. Нажмите /start чтобы начать заново.",
        reply_markup=main_menu_markup
    )
    return ConversationHandler.END


def main():
    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_main_menu)],
            CHATTING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message),
                CommandHandler("clear", clear_history_command),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_history_command))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()