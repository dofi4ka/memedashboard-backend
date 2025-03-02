from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import distinct
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.session import get_db
from app.models.post import Post, ProccedWeeks
from app.utils.advanced_logger import AdvancedLogger

week_router = APIRouter()
logger = AdvancedLogger("week_router")


@week_router.get("/weeks/{week_key}")
async def get_week(week_key: str, db: AsyncSession = Depends(get_db)):
    try:
        logger.info("Запрос на получение данных недели", week_key=week_key)

        query_week = select(ProccedWeeks).where(ProccedWeeks.week == week_key)
        result_week = await db.execute(query_week)
        week = result_week.scalars().first()

        if not week:
            logger.warning("Неделя не найдена", week_key=week_key)
            raise HTTPException(status_code=404, detail="Неделя не найдена")

        logger.info("Получение постов для недели", week_key=week_key)

        query_posts = select(Post).where(Post.week == week_key)
        result_posts = await db.execute(query_posts)
        posts = result_posts.scalars().all()

        posts_by_topic = defaultdict(list)
        for post in posts:
            posts_by_topic[post.topic].append(
                {
                    "id": str(post.id),
                    "text": post.text,
                    "description": post.description,
                    "photos": post.photos,
                    "likes": post.likes,
                    "views": post.views,
                    "date": post.date,
                    "url": post.url,
                }
            )

        response = {
            "week": week_key,
            "procced": week.procced,
            "posts_by_topic": dict(posts_by_topic),
        }

        logger.info("Данные недели успешно получены", week_key=week_key)
        return response

    except Exception as e:
        logger.error(
            "Ошибка при получении данных недели", week_key=week_key, error=str(e)
        )
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@week_router.get("/weeks")
async def get_weeks(db: AsyncSession = Depends(get_db)):
    try:
        logger.info("Запрос на получение списка доступных недель")

        query = select(distinct(ProccedWeeks.week))
        result = await db.execute(query)
        weeks = result.scalars().all()

        if not weeks:
            logger.info("Доступные недели не найдены")
            return {"weeks": []}

        logger.info("Список недель успешно получен", count=len(weeks))
        return {"weeks": weeks}

    except Exception as e:
        logger.error("Ошибка при получении списка недель", error=str(e))
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@week_router.get("/stats/weeks")
async def get_weeks_stats(db: AsyncSession = Depends(get_db)):
    try:
        logger.info("Запрос на получение статистики по неделям")

        weeks_query = select(distinct(ProccedWeeks.week))
        weeks_result = await db.execute(weeks_query)
        weeks = weeks_result.scalars().all()

        if not weeks:
            logger.info("Доступные недели не найдены")
            return {"stats": []}

        stats = []

        for week_key in weeks:
            topic_counts = defaultdict(int)

            posts_query = select(Post).where(Post.week == week_key)
            posts_result = await db.execute(posts_query)
            posts = posts_result.scalars().all()

            for post in posts:
                topic_counts[post.topic] += 1

            stats.append({"week": week_key, "topic_counts": topic_counts})

        logger.info("Статистика по неделям успешно получена", weeks_count=len(weeks))
        return {"stats": stats}

    except Exception as e:
        logger.error("Ошибка при получении статистики по неделям", error=str(e))
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")
