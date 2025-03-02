from uuid import UUID

from pydantic import BaseModel


class PostGet(BaseModel):
    id: UUID
    text: str
    description: str
    photos: list[str]
    likes: int
    views: int
    date: int
    week: str
    topic: str
