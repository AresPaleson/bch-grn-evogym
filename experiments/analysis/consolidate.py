import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, select

ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(ROOT))

from algorithms.EA_classes import GenerationSurvivor, Robot
from utils.body_metrics import BODY_METRICS, compute_body_metrics_from_genome
from utils.config import Config
from utils.metrics import METRICS_ABS, METRICS_REL

class Analysis:

    def __init__(self, args):
        self.study_name = args.study_name
        self.experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
        self.runs = [int(r) for r in args.runs.split(",") if r.strip()]
        self.final_gen = int(args.final_gen)
        self.path = f"{args.out_path}/{self.study_name}"
        self.max_voxels = args.max_voxels
        self.cube_face_size = args.cube_face_size
        self.voxel_types = args.voxel_types
        self.env_conditions = args.env_conditions
        self.plastic = args.plastic
        self.metrics = METRICS_ABS + METRICS_REL
        self.analysis_path = getattr(args, "analysis_dir", "") or f"{self.path}/analysis"

    def _resolve_db_path(self, base_path: str):
        if os.path.isdir(base_path):
            candidate = os.path.join(base_path, "experiment.sqlite3")
            return candidate if os.path.exists(candidate) else None
        return base_path if os.path.exists(base_path) else None

    def _compute_body_metric_frame(self, robots_df: pd.DataFrame) -> pd.DataFrame:
        if robots_df.empty:
            for metric in BODY_METRICS:
                robots_df[metric] = np.nan
            return robots_df
        if all(metric in robots_df.columns for metric in BODY_METRICS):
            missing_mask = robots_df[BODY_METRICS].isna().any(axis=1)
            if not missing_mask.any():
                return robots_df
        else:
            missing_mask = pd.Series(True, index=robots_df.index)
        computed_rows = []
        for row_idx, row in robots_df.loc[missing_mask].iterrows():
            metrics = compute_body_metrics_from_genome(
                row["genome"],
                max_voxels=self.max_voxels,
                cube_face_size=self.cube_face_size,
                voxel_types=self.voxel_types,
                env_conditions=self.env_conditions,
                plastic=self.plastic,
            )
            metrics["robot_id"] = row["robot_id"]
            metrics["_row_idx"] = row_idx
            computed_rows.append(metrics)
        if not computed_rows:
            return robots_df
        computed_df = pd.DataFrame(computed_rows)
        merged = robots_df.copy()
        computed_df = computed_df.set_index("_row_idx")
        for metric in BODY_METRICS:
            if metric not in merged.columns:
                merged[metric] = np.nan
            merged.loc[computed_df.index, metric] = computed_df[metric]
        return merged

    def consolidate(self):
        print("consolidating...")
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.analysis_path, exist_ok=True)
        frames = []
        robots_frames = []
        for experiment in self.experiments:
            for run in self.runs:
                db_base = os.path.join(self.path, experiment, f"run_{run}", f"run_{run}")
                db_path = self._resolve_db_path(db_base)
                if db_path is None:
                    print(f"[warn] DB not found, skipping: {db_base}")
                    continue
                engine = create_engine(f"sqlite:///{db_path}", future=True)
                inspector = inspect(engine)
                robot_columns = {col["name"] for col in inspector.get_columns("all_robots")}
                survivor_columns = {col["name"] for col in inspector.get_columns("generation_survivors")}
                available_abs_metrics = [m for m in METRICS_ABS if m in robot_columns]
                available_rel_metrics = [m for m in METRICS_REL if m in survivor_columns]
                with engine.connect() as conn:
                    survivors_stmt = select(
                        GenerationSurvivor.generation,
                        GenerationSurvivor.robot_id,
                        *(getattr(GenerationSurvivor, m) for m in available_rel_metrics),
                    )
                    df = pd.read_sql(survivors_stmt, conn)
                    robots_cols = [
                        Robot.robot_id,
                        Robot.born_generation,
                        Robot.num_voxels,
                        Robot.genome,
                    ]
                    for metric in available_abs_metrics:
                        if metric != "num_voxels":
                            robots_cols.append(getattr(Robot, metric))
                    robots_stmt = select(*robots_cols)
                    df_robots = pd.read_sql(robots_stmt, conn)
                df_robots = self._compute_body_metric_frame(df_robots)
                merge_cols = ["robot_id", "born_generation", "num_voxels"] + [
                    m for m in METRICS_ABS if m in df_robots.columns and m != "num_voxels"
                ]
                df = df.merge(df_robots[merge_cols], on="robot_id", how="left", validate="many_to_one")
                for metric in METRICS_REL:
                    if metric not in df.columns:
                        df[metric] = np.nan
                for metric in METRICS_ABS:
                    if metric not in df.columns:
                        df[metric] = np.nan
                df["experiment"] = experiment
                df["run"] = run
                frames.append(df)
                df_robots["experiment"] = experiment
                df_robots["run"] = run
                robots_frames.append(df_robots.drop(columns=["genome"]))
        if not frames:
            print("[warn] no data found; nothing to consolidate.")
            return
        all_df = pd.concat(frames, ignore_index=True)
        all_df = all_df[all_df["generation"] <= self.final_gen].reset_index(drop=True)
        all_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        all_df.to_csv(f"{self.analysis_path}/gens_robots.csv", index=False)
        if robots_frames:
            robots_all = pd.concat(robots_frames, ignore_index=True)
            robots_all.to_csv(f"{self.analysis_path}/all_robots.csv", index=False)
        else:
            print("[warn] no robots rows found.")
        active_metrics = [metric for metric in self.metrics if metric in all_df.columns and not all_df[metric].isna().all()]
        agg_dict = {}
        for metric in active_metrics:
            agg_dict[f"{metric}_mean"] = (metric, "mean")
            agg_dict[f"{metric}_std"] = (metric, lambda x: x.dropna().std(ddof=0))
            agg_dict[f"{metric}_max"] = (metric, "max")
        inner = (
            all_df.groupby(["experiment", "run", "generation"], as_index=False)
            .agg(**agg_dict)
        )
        inner.to_csv(f"{self.analysis_path}/gens_robots_inner.csv", index=False)
        agg_spec = {}
        for metric in active_metrics:
            mean_col = f"{metric}_mean"
            agg_spec[f"{mean_col}_mean"] = (mean_col, "mean")
            agg_spec[f"{mean_col}_std"] = (mean_col, lambda x: x.dropna().std(ddof=0))
            agg_spec[f"{mean_col}_median"] = (mean_col, "median")
            agg_spec[f"{mean_col}_q25"] = (mean_col, lambda x: x.dropna().quantile(0.25))
            agg_spec[f"{mean_col}_q75"] = (mean_col, lambda x: x.dropna().quantile(0.75))
            std_col = f"{metric}_std"
            agg_spec[f"{std_col}_mean"] = (std_col, "mean")
            agg_spec[f"{std_col}_median"] = (std_col, "median")
            agg_spec[f"{std_col}_q25"] = (std_col, lambda x: x.dropna().quantile(0.25))
            agg_spec[f"{std_col}_q75"] = (std_col, lambda x: x.dropna().quantile(0.75))
            max_col = f"{metric}_max"
            agg_spec[f"{max_col}_mean"] = (max_col, "mean")
            agg_spec[f"{max_col}_median"] = (max_col, "median")
            agg_spec[f"{max_col}_q25"] = (max_col, lambda x: x.dropna().quantile(0.25))
            agg_spec[f"{max_col}_q75"] = (max_col, lambda x: x.dropna().quantile(0.75))
            agg_spec[f"{max_col}_std"] = (max_col, lambda x: x.dropna().std(ddof=0))
        outer = inner.groupby(["experiment", "generation"], as_index=False).agg(**agg_spec)
        outer.to_csv(f"{self.analysis_path}/gens_robots_outer.csv", index=False)
        print("consolidated!")

if __name__ == "__main__":
    Analysis(Config()._get_params()).consolidate()
