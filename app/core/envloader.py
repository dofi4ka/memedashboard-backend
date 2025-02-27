import os

from dotenv import load_dotenv

load_dotenv()


class Environment:
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

    DEBUG: bool = os.getenv("DEBUG").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG" if DEBUG else "INFO")

    APP_TITLE: str = os.getenv("APP_TITLE", "FastAPI App")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = os.getenv("PORT", 8000)
    if PORT.isdigit():
        PORT = int(PORT)
    else:
        raise EnvironmentError("PORT должен быть числом")
