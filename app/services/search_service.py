import logging
import uuid
from typing import List, Tuple

from elasticsearch import AsyncElasticsearch
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.post import Post
from app.schemas.post import PostGet
from app.utils.advanced_logger import AdvancedLogger

logger = AdvancedLogger(__name__)


async def search_memes(
    db: AsyncSession, es_client: AsyncElasticsearch, query: str, limit: int, offset: int
) -> Tuple[List[PostGet], int]:
    try:
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "description"],
                    "fuzziness": "AUTO",
                }
            },
            "from": offset,
            "size": limit,
        }

        logger.debug("Выполняем поисковый запрос", search_body=search_body)
        response = await es_client.search(index="memes", body=search_body)

        total = response["hits"]["total"]["value"]
        logger.info("Найдено результатов", count=total, query=query)

        post_ids = [hit["_source"]["id"] for hit in response["hits"]["hits"]]

        if not post_ids:
            return [], total

        results = []
        for post_id in post_ids:
            try:
                if post_id == "None" or post_id is None:
                    logger.warning(
                        "Пропускаем недействительный post_id", post_id=post_id
                    )
                    continue

                uuid.UUID(post_id)

                stmt = select(Post).where(Post.id == post_id)
                result = await db.execute(stmt)
                post = result.scalar_one_or_none()

                if post:
                    results.append(
                        PostGet(
                            id=post.id,
                            text=post.text,
                            description=post.description,
                            photos=post.photos,
                            likes=post.likes,
                            views=post.views,
                            date=post.date,
                            week=post.week,
                            topic=post.topic,
                        )
                    )
            except (ValueError, TypeError) as e:
                logger.error(
                    "Ошибка при обработке post_id", post_id=post_id, error=str(e)
                )
                continue

        return results, total

    except Exception as e:
        logger.error("Ошибка при поиске мемов", error=str(e))
        raise
