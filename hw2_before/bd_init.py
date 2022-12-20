from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import yaml


with open("bd_config.yaml") as f:
    config = yaml.safe_load(f)
POSTGRES_HOST = config["POSTGRES_HOST"]
POSTGRES_DB = config["POSTGRES_DB"]
POSTGRES_USER = config["POSTGRES_USER"]
POSTGRES_PASSWORD = config["POSTGRES_PASSWORD"]

POSTGRES_CONN_STRING = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}"

engine_postgres = create_engine(POSTGRES_CONN_STRING)

Base = declarative_base()
SessionLocal = sessionmaker(bind=engine_postgres)
