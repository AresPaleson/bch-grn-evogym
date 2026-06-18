from sqlalchemy.orm import declarative_base
from sqlalchemy import (
    Column,
    Integer,
    Float,
    JSON,
    ForeignKey,
    PrimaryKeyConstraint,
)
from pathlib import Path
import sys

Base = declarative_base()

ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT))

from utils.metrics import METRICS_ABS, METRICS_REL


class ExperimentInfo(Base):
    __tablename__ = "experiment_info"
    id = Column(Integer, primary_key=True, autoincrement=True)
    seed = Column(Integer, nullable=False)

def build_robot_class():
    attrs = {
        "__tablename__": "all_robots",
        "robot_id": Column(Integer, primary_key=True),
        "born_generation": Column(Integer, nullable=False),
        "genome": Column(JSON, nullable=False),
        "valid": Column(Float, default=0.0),
        "parent1_id": Column(Integer, ForeignKey("all_robots.robot_id"), nullable=True),
        "parent2_id": Column(Integer, ForeignKey("all_robots.robot_id"), nullable=True),
    }
    for m in METRICS_ABS:
        attrs[m] = Column(Float)
    return type("Robot", (Base,), attrs)

Robot = build_robot_class()


def build_generation_survivor_class():
    attrs = {
        "__tablename__": "generation_survivors",
        "generation": Column(Integer, nullable=False),
        "robot_id": Column(Integer, ForeignKey("all_robots.robot_id"), nullable=False),
        "__table_args__": (
            PrimaryKeyConstraint("generation", "robot_id", name="pk_generation_robot"),
        ),
    }
    for m in METRICS_REL:
        attrs[m] = Column(Float, default=0.0)
    return type("GenerationSurvivor", (Base,), attrs)

GenerationSurvivor = build_generation_survivor_class()


class Individual:

    def __init__(self, genome, id_counter, parent1_id=None, parent2_id=None):
        self.id = id_counter
        self.genome = genome
        self.parent1_id = parent1_id
        self.parent2_id = parent2_id
        self.born_generation = None
        self.phenotype = None
        self.valid = 0
        self.evogym_structure = None
        self.evogym_connections = None
        self.evogym_actuator_meta = None
        self.evogym_controller = None
        for m in METRICS_ABS:
            if m == "displacement":
                setattr(self, m, float('-inf'))
            else:
                setattr(self, m, None)
        for m in METRICS_REL:
            setattr(self, m, None)
