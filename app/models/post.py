from uuid import uuid4

from sqlalchemy import ARRAY, Boolean, Column, Integer, String
from sqlalchemy.dialects.postgresql import UUID

from app.db.session import Base


class Post(Base):
    __tablename__ = "posts"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    week = Column(String)
    date = Column(Integer)
    text = Column(String)
    description = Column(String)
    photos = Column(ARRAY(String))
    url = Column(String)
    likes = Column(Integer)
    views = Column(Integer)
    topic = Column(String)


class ProccedWeeks(Base):
    __tablename__ = "procced_weeks"
    week = Column(String, primary_key=True)
    procced = Column(Boolean, default=False)
