import asyncio
import json
import logging
from collections import namedtuple
from datetime import datetime, timedelta
from io import BytesIO
from textwrap import dedent
from typing import Dict, List, Literal

import aiohttp
import imagehash
from elasticsearch import AsyncElasticsearch
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app import Environment
from app.models.post import Post, ProccedWeeks

logger = logging.getLogger(__name__)


class MemeAnalysis(BaseModel):
    is_meme: bool
    nsfw: bool
    description: str


class PostData(BaseModel):
    text: str
    photos: List[str]
    likes: int
    views: int
    date: int
    group_id: int
    post_url: str

    def get_likes_per_view(self) -> float:
        return self.likes / self.views if self.views else 0

    async def calculate_first_photo_hash(self) -> imagehash.ImageHash:
        async with aiohttp.ClientSession() as session:
            async with session.get(self.photos[0]) as response:
                content = await response.read()
                img = Image.open(BytesIO(content))
                return imagehash.phash(img)


class AnalyzedPost(PostData):
    analysis: MemeAnalysis


class WeeklyPostsCollection(BaseModel):
    posts: Dict[str, List[PostData]]


class AnalyzedWeeklyPostsCollection(BaseModel):
    posts: Dict[str, List[AnalyzedPost]]


async def is_attachments_valid(
    post: dict, max_photos_per_post: int
) -> list[str] | Literal[False]:
    """
    Проверяет, является ли пост интересным на основе его вложений.

    Args:
        post (dict): Словарь, представляющий пост.
        max_photos_per_post (int): Максимальное количество фотографий в посте.

    Returns:
        list[str]: Список URL-адресов фотографий, если пост интересный, иначе False.
    """
    if "attachments" not in post:
        logger.debug("Пост не имеет вложений")
        return False

    photos = [
        attach["photo"]["sizes"][-1]["url"]
        for attach in post["attachments"]
        if attach["type"] == "photo"
    ]
    if len(photos) > max_photos_per_post or len(photos) == 0:
        logger.debug(
            f"Пост имеет слишком много фото: {len(photos)} > {max_photos_per_post}"
        )
        return False

    if any(attach["type"] != "photo" for attach in post["attachments"]):
        logger.debug("Пост содержит не фото вложения")
        return False

    return photos


WeekTimestamp = namedtuple("WeekTimestamp", ["start", "end", "week_key"])


async def get_weekly_timestamps(weeks: int = 1) -> list[WeekTimestamp]:
    """
    Генерирует временные метки для последних завершенных недель.

    Args:
        weeks: Количество недель, для которых нужно сгенерировать
                              временные метки. По умолчанию 1.

    Returns:
        list[WeekTimestamp]: Список объектов WeekTimestamp, с отметками начала и конца недели.
    """
    now = datetime.now()

    days_since_monday = now.weekday()
    if days_since_monday == 0:
        last_completed_monday = now - timedelta(days=7)
    else:
        last_completed_monday = now - timedelta(days=days_since_monday + 7)

    last_completed_monday = last_completed_monday.replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    weekly_timestamps = []

    for i in range(weeks):
        week_start = last_completed_monday - timedelta(weeks=i)
        week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)

        weekly_timestamps.append(
            WeekTimestamp(
                start=int(week_start.timestamp()),
                end=int(week_end.timestamp()),
                week_key=f"{week_start.strftime('%Y-%m-%d')}",
            )
        )

    return weekly_timestamps


async def get_weekly_posts_with_images(
    access_token: str,
    group_ids: list[str],
    max_photos_per_post: int,
    weeks: int = 1,
    posts_per_week: int = 50,
    threshold: int = 5,
    weekly_timestamps: list[WeekTimestamp] = None,
) -> WeeklyPostsCollection:
    """
    Получает еженедельные посты с изображениями из указанных групп ВКонтакте.

    Args:
        access_token: Токен доступа к API ВКонтакте.
        group_ids: Список идентификаторов групп ВКонтакте.
        max_photos_per_post: Максимальное количество фотографий в посте.
        weeks: Количество недель для получения постов. По умолчанию 1.
        posts_per_week: Максимальное количество постов на неделю. По умолчанию 50.
        threshold: Пороговое значение для удаления дубликатов. По умолчанию 5.
    Returns:
        WeeklyPostsCollection: Словарь с ключами недель и списками постов с изображениями.
    """
    logger.info("Начинаем получение постов с изображениями")

    if not weekly_timestamps:
        weekly_timestamps = await get_weekly_timestamps(weeks)

    most_recent_timestamp = weekly_timestamps[0].end
    most_old_timestamp = weekly_timestamps[-1].start

    filtered_posts = []

    for group_id in group_ids:
        logger.info(f"Обработка группы с ID: {group_id}")
        page = 0
        while True:
            params = {
                "access_token": access_token,
                "owner_id": f"-{group_id}",
                "count": 100,
                "offset": page * 100,
                "extended": 1,
                "v": "5.131",
            }

            logger.debug(
                f"Выполняем API запрос для группы {group_id} на странице {page}"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.vk.com/method/wall.get", params=params
                ) as response:
                    if response.status != 200:
                        logger.error(
                            f"HTTP ошибка для группы {group_id}: Код {response.status}, {await response.text()}"
                        )
                        break

                    try:
                        data = await response.json()
                    except json.JSONDecodeError:
                        logger.error(
                            f"Ошибка декодирования JSON для группы {group_id}: {await response.text()}"
                        )
                        break

            if not data["response"].get("items"):
                logger.debug(f"Больше постов не найдено для группы {group_id}")
                break

            should_break = False

            posts = data["response"]["items"]

            for post in posts:
                if post["date"] < most_old_timestamp:
                    logger.debug(
                        f"Достигнуты посты после конечной даты для группы {group_id}"
                    )
                    should_break = True
                    break

                if post["date"] > most_recent_timestamp:
                    continue

                if photos := await is_attachments_valid(post, max_photos_per_post):
                    post_info = {
                        "text": post.get("text", ""),
                        "photos": photos,
                        "likes": post["likes"]["count"],
                        "views": post.get("views", {}).get("count", 0),
                        "date": post["date"],
                        "group_id": group_id,
                        "post_url": f"https://vk.com/wall-{group_id}_{post['id']}",
                    }
                    filtered_posts.append(PostData(**post_info))
                    logger.debug(
                        f"Добавлен пост из группы {group_id} с датой {post['date']}"
                    )

            if should_break:
                break

            page += 1

            if page >= 50:
                logger.warning(
                    f"Достигнут максимальный сдвиг (5000) для группы {group_id}"
                )
                break

    logger.info(
        f"Обработка завершена. Всего собрано постов: {len(filtered_posts)}, недель: {len(weekly_timestamps)}, постов в неделю: {len(filtered_posts) / len(weekly_timestamps)}"
    )

    filtered_posts.sort(key=lambda post: post.date)

    posts_by_weeks = WeeklyPostsCollection(
        posts={week_timestamp.week_key: [] for week_timestamp in weekly_timestamps}
    )
    for post in filtered_posts:
        for week_timestamp in weekly_timestamps:
            if week_timestamp.start <= post.date <= week_timestamp.end:
                posts_by_weeks.posts[week_timestamp.week_key].append(post)
                break

    return await remove_duplicates(
        await get_most_relevant_posts(posts_by_weeks, posts_per_week), threshold
    )


async def get_most_relevant_posts(
    weekly_posts: WeeklyPostsCollection, posts_per_week: int
) -> WeeklyPostsCollection:
    """
    Выбирает наиболее релевантные посты из коллекции на основе соотношения лайков к просмотрам.

    Args:
        posts: Словарь с ключами недель и списками постов с изображениями.
        posts_per_week: Максимальное количество постов, которое нужно выбрать для каждой недели.

    Returns:
        WeeklyPostsCollection: Словарь с отфильтрованными наиболее релевантными постами для каждой недели.
    """
    logger.info(f"Выбираем {posts_per_week} самых релевантных постов для каждой недели")

    most_relevant_posts = WeeklyPostsCollection(
        posts={week_key: [] for week_key in weekly_posts.posts.keys()}
    )
    for week_key, week_posts in weekly_posts.posts.items():
        sorted_posts = sorted(
            week_posts, key=lambda post: post.get_likes_per_view(), reverse=True
        )
        most_relevant_posts.posts[week_key] = sorted_posts[:posts_per_week]
        percentage = (
            (len(most_relevant_posts.posts[week_key]) / len(week_posts) * 100)
            if week_posts
            else 0
        )
        logger.debug(
            f"Неделя {week_key}: отобрано {len(most_relevant_posts.posts[week_key])} из {len(week_posts)} постов ({percentage:.2f}%)"
        )
    return most_relevant_posts


async def remove_duplicates(
    weekly_posts: WeeklyPostsCollection, threshold: float = 5
) -> WeeklyPostsCollection:
    """
    Удаляет дубликаты постов на основе хешей изображений.

    Args:
        posts: Словарь с ключами недель и списками постов с изображениями.
        threshold: Пороговое значение для удаления дубликатов.

    Returns:
        WeeklyPostsCollection: Словарь с отфильтрованными наиболее релевантными постами для каждой недели.
    """
    result = WeeklyPostsCollection(
        posts={week_key: [] for week_key in weekly_posts.posts.keys()}
    )

    for week, week_posts in weekly_posts.posts.items():
        logger.info(f"Удаление дубликатов в неделе {week}")

        post_hashes = {}
        for i, post in enumerate(week_posts):
            try:
                if not post.photos:
                    continue

                img_hash = await post.calculate_first_photo_hash()
                if img_hash:
                    post_hashes[i] = img_hash

            except Exception as e:
                logger.error(
                    f"Ошибка при обработке изображения из поста {post.post_url}: {e}"
                )

        processed_indices = set()
        merged_posts = []

        for i, hash1 in post_hashes.items():
            if i in processed_indices:
                continue

            processed_indices.add(i)
            similar_posts = [week_posts[i]]

            for j, hash2 in post_hashes.items():
                if i != j and j not in processed_indices:
                    if hash1 - hash2 < threshold:
                        similar_posts.append(week_posts[j])
                        processed_indices.add(j)

            if len(similar_posts) > 1:
                total_likes = sum(post.likes for post in similar_posts)
                total_views = sum(post.views for post in similar_posts)

                print(post.post_url for post in similar_posts)

                best_post = max(similar_posts, key=lambda p: p.likes)

                merged_post = PostData(
                    text=best_post.text,
                    photos=best_post.photos,
                    likes=total_likes,
                    views=total_views,
                    date=best_post.date,
                    group_id=best_post.group_id,
                    post_url=best_post.post_url,
                )

                merged_posts.append(merged_post)
                logger.debug(f"Объединено {len(similar_posts)} похожих постов в один")
            else:
                merged_posts.append(similar_posts[0])

        for i, post in enumerate(week_posts):
            if i not in processed_indices and i not in post_hashes:
                merged_posts.append(post)

        result.posts[week] = merged_posts
        logger.info(
            f"Неделя {week}: после удаления дубликатов осталось {len(merged_posts)} постов из {len(week_posts)}"
        )

    return result


async def analyze_post(post: PostData, openai_client: OpenAI) -> AnalyzedPost:
    content = [{"type": "text", "text": post.text}] + [
        {"type": "image_url", "image_url": {"url": photo, "detail": "low"}}
        for photo in post.photos
    ]

    logger.debug(f"Отправка запроса к API OpenAI для поста {post.post_url}")
    try:
        response = await asyncio.to_thread(
            openai_client.beta.chat.completions.parse,
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": dedent(
                        """
                    You are a helpful assistant that analyzes images and provides information about them.
                    is_meme: true if the image is a meme or a meme-like image, false if it news, report, advertisement or something else.
                    nsfw: true if the image has any uncensored bad words or explicit content, false otherwise.
                    description: very detailed. image, situation, humor, key elements. If you found some text, quote it in the description. If image can be related to some event write it in the description. Use provided text as context.
                    Answer in Russian.
                    """
                    ),
                },
                {"role": "user", "content": content},
            ],
            response_format=MemeAnalysis,
        )
        analysis = response.choices[0].message.parsed
        logger.info(f"Анализ поста {post.post_url}: {analysis}")
        logger.debug(
            f"Успешно обработан пост {post.post_url}. Использовано токенов: {response.usage.total_tokens}"
        )
        return AnalyzedPost(
            text=post.text,
            photos=post.photos,
            likes=post.likes,
            views=post.views,
            date=post.date,
            group_id=post.group_id,
            post_url=post.post_url,
            analysis=analysis,
        )
    except Exception as e:
        logger.error(
            f"Ошибка в запросе к API OpenAI для поста {post.post_url}: {str(e)}"
        )
        default_analysis = MemeAnalysis(
            is_meme=False,
            nsfw=False,
            description="Не удалось проанализировать изображение из-за ошибки API",
        )
        return AnalyzedPost(
            text=post.text,
            photos=post.photos,
            likes=post.likes,
            views=post.views,
            date=post.date,
            group_id=post.group_id,
            post_url=post.post_url,
            analysis=default_analysis,
        )


async def analyze_posts(posts: WeeklyPostsCollection) -> AnalyzedWeeklyPostsCollection:
    logger.debug("Начинаем анализ постов")

    logger.debug("Инициализируем OpenAI API")
    client = OpenAI()

    result = AnalyzedWeeklyPostsCollection(
        posts={week_key: [] for week_key in posts.posts.keys()}
    )

    for week, week_posts in posts.posts.items():
        filtered_posts = []
        tasks = []
        for post in week_posts:
            tasks.append(analyze_post(post, client))

        analyzed_posts = await asyncio.gather(*tasks, return_exceptions=True)

        for analyzed_post in analyzed_posts:
            if isinstance(analyzed_post, Exception):
                logger.error(f"Ошибка при анализе поста: {str(analyzed_post)}")
                continue

            try:
                if (
                    hasattr(analyzed_post, "analysis")
                    and analyzed_post.analysis.is_meme
                    and not analyzed_post.analysis.nsfw
                ):
                    filtered_posts.append(analyzed_post)
                    logger.debug(
                        f"Добавлен мем {analyzed_post.post_url} в отфильтрованные посты"
                    )
                else:
                    logger.debug(
                        f"Пост {analyzed_post.post_url} не прошел фильтрацию: не мем или NSFW"
                    )
            except Exception as e:
                logger.error(f"Ошибка при обработке результата анализа: {str(e)}")

        result.posts[week] = filtered_posts
        logger.info(
            f"Неделя {week}: отфильтровано {len(filtered_posts)} мемов из {len(week_posts)} постов"
        )

    logger.debug("Анализ постов завершен")
    return result


async def parse_posts(session: AsyncSession, es_client: AsyncElasticsearch):
    weekly_timestamps = await get_weekly_timestamps(Environment.PARSE_WEEKS_COUNT)
    weekly_timestamps_to_parse = []
    for week_timestamp in weekly_timestamps:
        existing_week = await session.get(ProccedWeeks, week_timestamp.week_key)
        if existing_week and existing_week.procced:
            logger.info(
                f"Неделя {week_timestamp.week_key} уже была обработана ранее, пропускаем"
            )
            continue
        weekly_timestamps_to_parse.append(week_timestamp)

    if not weekly_timestamps_to_parse:
        logger.info("Нечего парсить, все недели уже были обработаны")
        return

    posts = await get_weekly_posts_with_images(
        group_ids=Environment.VK_GROUP_IDS,
        access_token=Environment.VK_ACCESS_TOKEN,
        max_photos_per_post=Environment.PARSE_MAX_PHOTOS_PER_POST,
        weeks=Environment.PARSE_WEEKS_COUNT,
        posts_per_week=Environment.PARSE_POSTS_PER_WEEK,
        threshold=5,
        weekly_timestamps=weekly_timestamps_to_parse,
    )
    analyzed_posts = await analyze_posts(posts)
    
    for week, week_posts in analyzed_posts.posts.items():
        logger.info(f"Обработка постов для недели {week}")
        procced_week = ProccedWeeks(week=week, procced=True)
        
        session.add(procced_week)
        
        for post in week_posts:
            db_post = Post(
                week=week,
                text=post.text,
                description=post.analysis.description,
                photos=post.photos,
                url=post.post_url,
                likes=post.likes,
                views=post.views,
                date=post.date
            )
            
            session.add(db_post)
            await session.commit()
            
            await es_client.index(index="memes", body={
                "id": str(db_post.id),
                "title": post.text,
                "description": post.analysis.description
            })


if __name__ == "__main__":
    posts = get_weekly_posts_with_images(
        access_token=Environment.VK_ACCESS_TOKEN,
        group_ids=Environment.VK_GROUP_IDS,
        max_photos_per_post=Environment.PARSE_MAX_PHOTOS_PER_POST,
        weeks=Environment.PARSE_WEEKS_COUNT,
        posts_per_week=Environment.PARSE_POSTS_PER_WEEK,
    )

    analyzed_posts = analyze_posts(posts)

    with open("analyzed_posts.json", "w", encoding="utf-8") as f:
        json.dump(analyzed_posts.model_dump(), f, ensure_ascii=False, indent=4)

    logger.info(f"Анализ завершен. Результаты сохранены в analyzed_posts.json")
