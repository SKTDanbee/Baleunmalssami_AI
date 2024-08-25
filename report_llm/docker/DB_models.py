from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean

DB_url = os.getenv('DB_URL')
print(DB_url)
try:
    engine = create_engine(DB_url, pool_recycle=500, echo=True)
    print("DB 연결 성공")
except Exception as e:
    print(f"데이터베이스 연결 중 오류 발생: {e}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Child(Base):
    __tablename__ = 'child'

    id = Column(String(255), primary_key=True)
    password = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    parent_phone_number = Column(String(255), nullable=True)
    is_verified = Column(Boolean, default=False)
    parent_id = Column(String(255), ForeignKey('parent.id'))
    parent = relationship('Parent', back_populates='children')


class Parent(Base):
    __tablename__ = 'parent'

    id = Column(String(255), primary_key=True)
    password = Column(String(255), nullable=True)
    name = Column(String(255), nullable=True)
    phone_number = Column(String(255),unique=True, nullable=True)
    verification_code = Column(String(255), nullable=True)
    children = relationship('Child', back_populates='parent')

class Report(Base):
    __tablename__ = 'report'

    id = Column(Integer, primary_key=True)
    report_date = Column(DateTime, nullable=False)
    report = Column(Text, nullable=False)
    child_id = Column(String(255), nullable=False)
    abuse_count = Column(Integer, nullable=False)