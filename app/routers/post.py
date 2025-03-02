from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from uuid import UUID

from app.db.session import get_db
from app.models.post import Post
from app.utils.advanced_logger import AdvancedLogger

post_router = APIRouter()
logger = AdvancedLogger("post_router")


@post_router.get("/post/{post_id}")
async def get_post(post_id: UUID, db: AsyncSession = Depends(get_db)):
    try:
        logger.info("Запрос на получение поста", post_id=post_id)

        query = select(Post).where(Post.id == post_id)
        result = await db.execute(query)
        post = result.scalars().first()

        if not post:
            logger.warning("Пост не найден", post_id=post_id)
            raise HTTPException(status_code=404, detail="Пост не найден")

        logger.info("Пост успешно получен", post_id=post_id)
        return post
    except Exception as e:
        logger.error("Ошибка при получении поста", post_id=post_id, error=str(e))
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")
