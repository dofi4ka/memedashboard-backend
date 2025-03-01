import logging
from typing import List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from elasticsearch import AsyncElasticsearch
from app.models.post import Post
from app.schemas.post import PostGet

logger = logging.getLogger(__name__)

async def search_memes(
    db: AsyncSession,
    es_client: AsyncElasticsearch,
    query: str,
    limit: int,
    offset: int
) -> Tuple[List[PostGet], int]:
    try:
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "description"],
                    "fuzziness": "AUTO"
                }
            },
            "from": offset,
            "size": limit
        }
        
        logger.debug(f"Выполняем поисковый запрос: {search_body}")
        response = await es_client.search(index="memes", body=search_body)
        
        total = response["hits"]["total"]["value"]
        logger.info(f"Найдено {total} результатов для запроса '{query}'")
        
        post_ids = [hit["_source"]["id"] for hit in response["hits"]["hits"]]
        
        logger.info(f"Найдены посты: {post_ids}")
        if not post_ids:
            return [], total
            
        results = []
        for post_id in post_ids:
            post = await db.get(Post, post_id)
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
                        week=post.week
                    )
                )
            
        return results, total
        
    except Exception as e:
        logger.error(f"Ошибка при поиске мемов: {str(e)}")
        raise
