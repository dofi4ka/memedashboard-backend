from elasticsearch import AsyncElasticsearch
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app import Environment
from app.utils.advanced_logger import AdvancedLogger

engine = create_async_engine(Environment.DATABASE_URL, pool_pre_ping=True)
AsyncSessionLocal = sessionmaker(
    class_=AsyncSession, autocommit=False, autoflush=False, bind=engine
)


logger = AdvancedLogger(__name__)

metadata = MetaData()
Base = declarative_base(metadata=metadata)


async def init_db(session: AsyncSession):
    """
    Инициализирует базу данных и создает таблицы, если они не существуют.
    """
    try:
        from importlib import import_module

        import_module("app.models.post")

        async with engine.begin() as conn:
            await conn.run_sync(lambda conn: metadata.create_all(conn))

        logger.info("База данных инициализирована успешно")
    except Exception as e:
        logger.error("Ошибка при инициализации базы данных", error=str(e))
        raise


def get_db():
    """
    Возвращает асинхронный генератор сессии базы данных.
    Использует глобальный AsyncSessionLocal.
    """

    async def _get_db():
        async with AsyncSessionLocal() as db:
            try:
                yield db
            finally:
                await db.close()

    return _get_db


async def get_es_client():
    return AsyncElasticsearch(Environment.ELASTICSEARCH_URL)


async def init_elasticsearch(es_client: AsyncElasticsearch):
    """
    Инициализирует индексы Elasticsearch, если они не существуют.
    """
    try:
        # Проверяем, существует ли индекс
        index_exists = await es_client.indices.exists(index="memes")

        if not index_exists:
            # Настройки индекса
            settings = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "russian_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "russian_stemmer"],
                            }
                        },
                        "filter": {
                            "russian_stemmer": {
                                "type": "stemmer",
                                "language": "russian",
                            }
                        },
                    },
                },
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "title": {"type": "text", "analyzer": "russian_analyzer"},
                        "description": {"type": "text", "analyzer": "russian_analyzer"},
                    }
                },
            }

            await es_client.indices.create(index="memes", body=settings)
            logger.info("Индекс 'memes' в Elasticsearch создан успешно")
        else:
            logger.info("Индекс 'memes' в Elasticsearch уже существует")

    except Exception as e:
        logger.error("Ошибка при инициализации Elasticsearch", error=str(e))
        raise
