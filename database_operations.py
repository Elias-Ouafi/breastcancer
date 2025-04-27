from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd

# Database connection
DATABASE_URL = "postgresql://postgres:password@localhost/breast_cancer_db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class PatientData(Base):
    __tablename__ = 'patient_data'
    
    id = Column(Integer, primary_key=True)
    radius_mean = Column(Float)
    texture_mean = Column(Float)
    perimeter_mean = Column(Float)
    area_mean = Column(Float)
    smoothness_mean = Column(Float)
    compactness_mean = Column(Float)
    concavity_mean = Column(Float)
    concave_points_mean = Column(Float)
    symmetry_mean = Column(Float)
    fractal_dimension_mean = Column(Float)
    diagnosis = Column(String(1))
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelResults(Base):
    __tablename__ = 'model_results'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

def store_patient_data(data):
    """Store patient data in the database."""
    session = Session()
    try:
        patient = PatientData(**data)
        session.add(patient)
        session.commit()
        return patient.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def store_model_results(results):
    """Store model results in the database."""
    session = Session()
    try:
        for model_name, metrics in results.items():
            model_result = ModelResults(
                model_name=model_name,
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1'],
                roc_auc=metrics['roc_auc']
            )
            session.add(model_result)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_patient_data():
    """Retrieve all patient data from the database."""
    session = Session()
    try:
        query = session.query(PatientData)
        return pd.read_sql(query.statement, session.bind)
    finally:
        session.close()

def get_model_results():
    """Retrieve all model results from the database."""
    session = Session()
    try:
        query = session.query(ModelResults)
        return pd.read_sql(query.statement, session.bind)
    finally:
        session.close()

def get_best_model():
    """Retrieve the best performing model based on accuracy."""
    session = Session()
    try:
        query = session.query(ModelResults).order_by(ModelResults.accuracy.desc()).first()
        return query
    finally:
        session.close()

if __name__ == "__main__":
    # Create tables
    Base.metadata.create_all(engine) 