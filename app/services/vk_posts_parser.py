import asyncio
import json
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
from app.schemas.topics import topics, validate_topic
from app.utils.advanced_logger import AdvancedLogger

logger = AdvancedLogger(__name__)


class MemeAnalysis(BaseModel):
    is_meme: bool
    nsfw: bool
    description: str
    topic: str


class WeeklyAnalysis(BaseModel):
    class Topic(BaseModel):
        name: str
        description: str
        post_urls: List[str]

    topics: List[Topic]
    summary: str


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
            "Пост имеет слишком много фото",
            photos_count=len(photos),
            max_photos=max_photos_per_post,
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
        logger.info("Обработка группы", group_id=group_id)
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
                "Выполняем API запрос для группы", group_id=group_id, page=page
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.vk.com/method/wall.get", params=params
                ) as response:
                    if response.status != 200:
                        logger.error(
                            "HTTP ошибка для группы",
                            group_id=group_id,
                            status_code=response.status,
                            response_text=await response.text(),
                        )
                        break

                    try:
                        data = await response.json()
                    except json.JSONDecodeError:
                        logger.error(
                            "Ошибка декодирования JSON для группы",
                            group_id=group_id,
                            response_text=await response.text(),
                        )
                        break

            if not data["response"].get("items"):
                logger.debug("Больше постов не найдено для группы", group_id=group_id)
                break

            should_break = False

            posts = data["response"]["items"]

            for post in posts:
                if post["date"] < most_old_timestamp:
                    logger.debug(
                        "Достигнуты посты после конечной даты для группы",
                        group_id=group_id,
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
                        "Добавлен пост из группы", group_id=group_id, date=post["date"]
                    )

            if should_break:
                break

            page += 1

            if page >= 50:
                logger.warning(
                    "Достигнут максимальный сдвиг (5000) для группы", group_id=group_id
                )
                break

    logger.info(
        "Обработка завершена",
        total_posts=len(filtered_posts),
        weeks_count=len(weekly_timestamps),
        posts_per_week=len(filtered_posts) / len(weekly_timestamps),
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
    logger.info(
        "Выбираем самых релевантных постов для каждой недели",
        posts_per_week=posts_per_week,
    )

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
            "Неделя: отобрано постов",
            week=week_key,
            selected_posts=len(most_relevant_posts.posts[week_key]),
            total_posts=len(week_posts),
            percentage=f"{percentage:.2f}%",
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
        logger.info("Удаление дубликатов в неделе", week=week)

        post_hashes = {}

        async def process_post(i, post):
            try:
                if not post.photos:
                    return None

                img_hash = await post.calculate_first_photo_hash()
                if img_hash:
                    return (i, img_hash)
                return None
            except Exception as e:
                logger.error(
                    "Ошибка при обработке изображения из поста",
                    post_url=post.post_url,
                    error=str(e),
                )
                return None

        tasks = [process_post(i, post) for i, post in enumerate(week_posts)]
        task_results = await asyncio.gather(*tasks)

        for task_result in task_results:
            if task_result:
                i, img_hash = task_result
                post_hashes[i] = img_hash

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
                logger.debug(
                    "Объединено похожих постов в один", count=len(similar_posts)
                )
            else:
                merged_posts.append(similar_posts[0])

        for i, post in enumerate(week_posts):
            if i not in processed_indices and i not in post_hashes:
                merged_posts.append(post)

        result.posts[week] = merged_posts
        logger.info(
            "Неделя: после удаления дубликатов",
            week=week,
            remaining_posts=len(merged_posts),
            original_posts=len(week_posts),
        )

    return result


async def analyze_post(post: PostData, openai_client: OpenAI) -> AnalyzedPost:
    content = [{"type": "text", "text": post.text}] + [
        {"type": "image_url", "image_url": {"url": photo, "detail": "low"}}
        for photo in post.photos
    ]

    logger.debug("Отправка запроса к API OpenAI для поста", post_url=post.post_url)
    try:
        response = await asyncio.to_thread(
            openai_client.beta.chat.completions.parse,
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "developer",
                    "content": dedent(
                        f"""
                    You are a helpful assistant that analyzes images and provides detailed information about them.
                    is_meme: true if the image is a meme or a meme-like image, false if it is a news image, report, advertisement, or something else.
                    nsfw: true if the image contains any uncensored bad language or explicit content, false otherwise.
                    description: Provide a very detailed description of the image including the depicted scene, situation, humor, and key elements. If there is any text present in the image, quote it in the description. If the image can be related to a specific event, mention it. Use the provided text as context.
                    topic: Determine the most appropriate topic for the meme. The topic must be chosen strictly from the following list:
                    {", ".join([f"- {topic.name}: {topic.chatgpt_description}" for topic in topics.values()])}

                    Strictly choose one topic from the list above, if there is no topic that can be related to the meme, select "случайные".

                    Answer in Russian.
                    """
                    ),
                },
                {"role": "user", "content": content},
            ],
            response_format=MemeAnalysis,
        )
        analysis = response.choices[0].message.parsed
        analysis.topic = validate_topic(analysis.topic)
        logger.info("Анализ поста", post_url=post.post_url, analysis=analysis)
        logger.debug(
            "Успешно обработан пост",
            post_url=post.post_url,
            tokens_used=response.usage.total_tokens,
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
            "Ошибка в запросе к API OpenAI для поста",
            post_url=post.post_url,
            error=str(e),
        )
        default_analysis = MemeAnalysis(
            is_meme=False,
            nsfw=False,
            description="Не удалось проанализировать изображение из-за ошибки API",
            topic=validate_topic("случайные"),
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


async def analyze_posts(
    posts: WeeklyPostsCollection, openai_client: OpenAI
) -> AnalyzedWeeklyPostsCollection:
    logger.debug("Начинаем анализ постов")

    result = AnalyzedWeeklyPostsCollection(
        posts={week_key: [] for week_key in posts.posts.keys()}
    )

    for week, week_posts in posts.posts.items():
        filtered_posts = []
        tasks = []
        for post in week_posts:
            tasks.append(analyze_post(post, openai_client))

        analyzed_posts = await asyncio.gather(*tasks, return_exceptions=True)

        for analyzed_post in analyzed_posts:
            if isinstance(analyzed_post, Exception):
                logger.error("Ошибка при анализе поста", error=str(analyzed_post))
                continue

            try:
                if (
                    hasattr(analyzed_post, "analysis")
                    and analyzed_post.analysis.is_meme
                    and not analyzed_post.analysis.nsfw
                ):
                    filtered_posts.append(analyzed_post)
                    logger.debug(
                        "Добавлен мем в отфильтрованные посты",
                        post_url=analyzed_post.post_url,
                    )
                else:
                    logger.debug(
                        "Пост не прошел фильтрацию: не мем или NSFW",
                        post_url=analyzed_post.post_url,
                    )
            except Exception as e:
                logger.error("Ошибка при обработке результата анализа", error=str(e))

        result.posts[week] = filtered_posts
        logger.info(
            "Неделя: отфильтровано мемов",
            week=week,
            memes_count=len(filtered_posts),
            total_posts=len(week_posts),
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
                "Неделя уже была обработана ранее, пропускаем",
                week=week_timestamp.week_key,
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

    logger.debug("Инициализируем OpenAI API")
    openai_client = OpenAI()

    analyzed_posts = await analyze_posts(posts, openai_client)

    for week, week_posts in analyzed_posts.posts.items():
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
                date=post.date,
                topic=post.analysis.topic,
            )

            session.add(db_post)
            await session.commit()

            await es_client.index(
                index="memes",
                body={
                    "id": str(db_post.id),
                    "title": post.text,
                    "description": post.analysis.description,
                },
            )


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

    logger.info("Анализ завершен. Результаты сохранены в analyzed_posts.json")
