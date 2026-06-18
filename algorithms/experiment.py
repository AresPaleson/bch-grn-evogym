import os, sys
import random
import sqlite3
from sqlalchemy import create_engine, func, event, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT))

from algorithms.EA_classes import Base, Robot, GenerationSurvivor, Individual, ExperimentInfo
from utils.metrics import METRICS_ABS, METRICS_REL

@event.listens_for(Engine, "connect")

def _set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

class Experiment:
    """
    Handles experiment bookkeeping:
      - output/db paths
      - SQLAlchemy engine/session
      - RNG seed management
      - state recovery from DB
      - atomic persistence per generation
    """

    def __init__(self, args):
        self.out_path = f"{args.out_path}/{args.study_name}/{args.experiment_name}/run_{args.run}"
        os.makedirs(self.out_path, exist_ok=True)
        self.db_path = os.path.join(self.out_path, f'run_{args.run}')
        self.voxel_types = args.voxel_types

    def recover_db(self):
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self._migrate_sqlite_schema()
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.session = self.Session()
        self.rng = random.Random()
        info = self.session.query(ExperimentInfo).first()
        if info is None:
            seed = random.randint(0, 2**32 - 1)
            print("seed (new)", seed)
            self.rng.seed(seed)
            self.session.add(ExperimentInfo(seed=seed))
            self.session.commit()
        else:
            print("seed (reused)", info.seed)
            self.rng.seed(info.seed)
        self.id_counter = 0

    def _migrate_sqlite_schema(self):
        inspector = inspect(self.engine)
        if "all_robots" in inspector.get_table_names():
            existing_robot_cols = {col["name"] for col in inspector.get_columns("all_robots")}
            missing_robot_metrics = [m for m in METRICS_ABS if m not in existing_robot_cols]
            if missing_robot_metrics:
                with self.engine.begin() as conn:
                    for metric in missing_robot_metrics:
                        conn.exec_driver_sql(f"ALTER TABLE all_robots ADD COLUMN {metric} FLOAT")
        if "generation_survivors" in inspector.get_table_names():
            existing_survivor_cols = {col["name"] for col in inspector.get_columns("generation_survivors")}
            missing_survivor_metrics = [m for m in METRICS_REL if m not in existing_survivor_cols]
            if missing_survivor_metrics:
                with self.engine.begin() as conn:
                    for metric in missing_survivor_metrics:
                        conn.exec_driver_sql(f"ALTER TABLE generation_survivors ADD COLUMN {metric} FLOAT")

    def _individual_from_robot(self, r: Robot) -> Individual:
        ind = Individual(genome=r.genome, id_counter=r.robot_id,
                         parent1_id=r.parent1_id, parent2_id=r.parent2_id)
        ind.valid = r.valid
        ind.born_generation = r.born_generation
        for m in METRICS_ABS:
            setattr(ind, m, getattr(r, m, None))
        for m in METRICS_REL:
            setattr(ind, m, getattr(r, m, None))
        return ind

    def _recover_state(self):
        """
        Returns (last_completed_generation, recovered_population or None).
        If there is no completed generation, returns (None, None).
        Requires subclass to implement `develop_phenotype(genome)`.
        """
        with self.Session() as s:
            last_gen = s.query(func.max(GenerationSurvivor.generation)).scalar()
            if last_gen is None:
                if s.query(Robot).count() != 0:
                    raise RuntimeError(
                        "DB inconsistent: robots exist but no survivors. Clean or migrate."
                    )
                self.id_counter = 0
                return None, None
            rows = (
                s.query(Robot, GenerationSurvivor)
                .join(GenerationSurvivor, GenerationSurvivor.robot_id == Robot.robot_id)
                .filter(GenerationSurvivor.generation == last_gen)
                .all()
            )
            population = []
            for r, gs in rows:
                ind = self._individual_from_robot(r)
                ind.phenotype = self.develop_phenotype(ind.genome, self.voxel_types)
                for m in METRICS_REL:
                    setattr(ind, m, getattr(gs, m, None))
                population.append(ind)
            max_id = s.query(func.max(Robot.robot_id)).scalar()
            self.id_counter = int(max_id) if max_id is not None else 0
            return int(last_gen), population

    def _persist_generation_atomic(self, generation, robots_this_gen, survivors_this_gen):
        with self.Session() as s, s.begin():
            for ind in robots_this_gen:
                self._stage_robot(s, ind)
            s.flush()
            self._stage_generation_survivors(s, generation, survivors_this_gen)

    def _stage_robot(self, s, individual):
        row = s.get(Robot, individual.id)
        if row is None:
            data = {
                "robot_id": individual.id,
                "born_generation": int(individual.born_generation),
                "genome": individual.genome,
                "valid": individual.valid,
                "parent1_id": individual.parent1_id,
                "parent2_id": individual.parent2_id,
            }
            for m in METRICS_ABS:
                data[m] = getattr(individual, m, None)
            s.add(Robot(**data))

    def _stage_generation_survivors(self, s, generation, survivors):
        for ind in survivors:
            data = {
                "generation": int(generation),
                "robot_id": int(ind.id),
            }
            for m in METRICS_REL:
                data[m] = getattr(ind, m, None)
            s.merge(GenerationSurvivor(**data))
