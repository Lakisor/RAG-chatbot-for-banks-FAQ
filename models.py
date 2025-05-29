from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
import json

Base = declarative_base()

class FAQ(Base):
    __tablename__ = 'faqs'
    
    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    embedding = Column(Text)  # Will store JSON-serialized vector
    
    def set_embedding(self, vector):
        self.embedding = json.dumps(vector.tolist() if hasattr(vector, 'tolist') else vector)
    
    def get_embedding(self):
        return np.array(json.loads(self.embedding)) if self.embedding else None

class Database:
    def __init__(self, db_path='faq_database.db'):
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        return self.Session()
