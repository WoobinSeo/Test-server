"""
DB 연결 및 공통 설정 로딩 유틸리티.

- `.env` 에서 DB 접속 정보 로드
- `config.yaml` 에서 종목 리스트 로드
"""

import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from sqlalchemy import create_engine
import yaml


def get_engine():
    """환경변수(.env)를 사용해 SQLAlchemy 엔진 생성."""
    load_dotenv()
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")
    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    return create_engine(conn_str, echo=False, future=True)


def load_config_stocks(config_path: str = "config.yaml") -> List[Dict[str, Any]]:
    """`config.yaml` 에서 종목 설정 리스트 로드."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["stocks"]








