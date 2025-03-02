from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import distinct

from app.db.session import get_db
from app.models.post import Post
from app.utils.advanced_logger import AdvancedLogger
from app.schemas.topics import topics
topics_router = APIRouter()
logger = AdvancedLogger("topics_router")


@topics_router.get("/topics")
async def get_topics():
    return {"topics": [{"name": topic.name, "description": topic.user_description} for topic in topics.values()]}