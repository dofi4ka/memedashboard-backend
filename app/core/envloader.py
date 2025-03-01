import os

from dotenv import load_dotenv

load_dotenv()


class Environment:
    VK_ACCESS_TOKEN: str = os.getenv("VK_ACCESS_TOKEN")
    VK_GROUP_IDS: list[str] = os.getenv("VK_GROUP_IDS", "").split(",")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    if not VK_ACCESS_TOKEN:
        raise EnvironmentError("VK_ACCESS_TOKEN не задан")
    if not VK_GROUP_IDS:
        raise EnvironmentError("VK_GROUP_IDS не задан")
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY не задан")

    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST")
    POSTGRES_PORT: int = os.getenv("POSTGRES_PORT")
    DATABASE_URL: str = os.getenv("DATABASE_URL")

    if not DATABASE_URL:
        if not all(
            [
                POSTGRES_USER,
                POSTGRES_PASSWORD,
                POSTGRES_HOST,
                POSTGRES_PORT,
                POSTGRES_DB,
            ]
        ):
            raise EnvironmentError(
                "Не заданы обязательные переменные окружения для подключения к базе данных"
            )
        DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG" if DEBUG else "INFO")

    APP_TITLE: str = os.getenv("APP_TITLE", "FastAPI App")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = os.getenv("PORT", "8000")
    if PORT.isdigit():
        PORT = int(PORT)
    else:
        raise EnvironmentError("PORT должен быть числом")
