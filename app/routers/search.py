import logging
from typing import List, Optional

from elasticsearch import AsyncElasticsearch
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db, get_es_client
from app.schemas.post import PostGet
from app.services.search_service import search_memes

search_router = APIRouter()

logger = logging.getLogger(__name__)


@search_router.get("/search", response_model=list[PostGet])
async def search_endpoint(
    query: str = Query(..., description="Поисковый запрос"),
    page: int = Query(1, description="Номер страницы", ge=1),
    page_size: int = Query(10, description="Размер страницы", ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    es_client: AsyncElasticsearch = Depends(get_es_client),
):
    offset = (page - 1) * page_size
    results, total = await search_memes(db, es_client, query, page_size, offset)
    logger.info(f"Найдено {total} результатов для запроса '{query}'")
    return results
