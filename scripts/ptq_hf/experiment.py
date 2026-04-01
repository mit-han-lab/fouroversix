from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Experiment(Base):
    """A PTQ experiment with results."""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=func.now())
    hostname = Column(String)
    group_name = Column(String)
    model_name = Column(String, nullable=False)
    task = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    ptq_method = Column(String, nullable=False)
    scale_rule = Column(String)
    activation_scale_rule = Column(String)
    weight_scale_rule = Column(String)
    dtype = Column(String)
    activation_dtype = Column(String)
    weight_dtype = Column(String)
    smoothquant_alpha = Column(Float, nullable=True)
    results = Column(JSON, nullable=False)
