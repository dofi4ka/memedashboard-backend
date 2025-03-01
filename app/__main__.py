import uvicorn
from fastapi import FastAPI

from app import Environment
from app.core.logging import configure_logging
from app.db.session import init_db, get_db, get_es_client, init_elasticsearch
from app.routers.search import search_router
from app.routers.post import post_router
from app.utils.advanced_logger import AdvancedLogger

configure_logging()

from contextlib import asynccontextmanager

from elasticsearch import AsyncElasticsearch
from sqlalchemy.ext.asyncio import (AsyncSession, async_sessionmaker,
                                    create_async_engine)

from app.services.vk_posts_parser import parse_posts

logger = AdvancedLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация баз данных
    engine = create_async_engine(
        Environment.DATABASE_URL, 
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=Environment.DEBUG
    )
    
    async_session = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )
    
    es_client = AsyncElasticsearch(
        Environment.ELASTICSEARCH_URL,
        retry_on_timeout=True,
        max_retries=5
    )
    
    # Сохраняем зависимости в приложении
    app.state.db_session = async_session
    app.state.es_client = es_client
    
    # Предоставляем зависимости для роутов
    app.dependency_overrides[get_db] = get_db()
    app.dependency_overrides[get_es_client] = lambda: es_client
    
    logger.info("Приложение запущено, зависимости инициализированы")
    
    try:
        # Инициализация базы данных
        async with async_session() as session:
            await init_db(session)
        
        # Инициализация Elasticsearch
        await init_elasticsearch(es_client)
        
        # Запуск парсинга в фоновом режиме только если включен DEBUG
        if Environment.DEBUG:
            async with async_session() as session:
                await parse_posts(session, es_client)
        
        yield
    finally:
        await engine.dispose()
        await es_client.close()
        logger.info("Приложение остановлено, ресурсы освобождены")


app = FastAPI(title=Environment.APP_TITLE, debug=Environment.DEBUG, lifespan=lifespan, root_path="/api")

app.include_router(search_router)
app.include_router(post_router)
uvicorn.run(app, host=Environment.HOST, port=Environment.PORT, log_config=None)
