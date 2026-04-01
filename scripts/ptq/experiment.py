from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Experiment(Base):
    """A PTQ experiment with results."""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=func.now())
    repeat = Column(Integer, nullable=False)
    hostname = Column(String, nullable=False)
    slurm_job_id = Column(Integer)
    log_path = Column(String)
    model = Column(String, nullable=False)
    task = Column(String, nullable=False)
    rht = Column(Boolean, nullable=False)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    ptq_method = Column(String, nullable=False)
    quantization_scheme = Column(String)
    results = Column(JSON, nullable=False)
