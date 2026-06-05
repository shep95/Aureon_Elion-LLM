"""SQLAlchemy engine and session management."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base

_engine = None
_SessionLocal = None

DEFAULT_SQLITE = "sqlite:///./data/aureon.db"


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL", DEFAULT_SQLITE)
    # Railway/Heroku use postgres:// — SQLAlchemy 2 needs postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        url = get_database_url()
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def init_db() -> None:
    from pathlib import Path

    url = get_database_url()
    if url.startswith("sqlite"):
        Path("data").mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    if _SessionLocal is None:
        get_engine()
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def db_available() -> bool:
    return bool(os.environ.get("DATABASE_URL")) or True  # sqlite always available locally
