from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import Environment

engine = create_engine(Environment.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
