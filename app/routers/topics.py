from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import distinct
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.session import get_db
from app.models.post import Post
from app.schemas.topics import topics
from app.utils.advanced_logger import AdvancedLogger

topics_router = APIRouter()
logger = AdvancedLogger("topics_router")


@topics_router.get("/topics")
async def get_topics():
    return {
        "topics": [
            {"name": topic.name, "description": topic.user_description}
            for topic in topics.values()
        ]
    }
