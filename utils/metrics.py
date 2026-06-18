import numpy as np
import sys
from pathlib import Path
from sklearn.neighbors import KDTree

ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT))

from algorithms.voxel_types import VOXEL_TYPES, VOXEL_TYPES_NOBONE
from utils.body_metrics import BODY_METRICS, compute_body_metrics_from_phenotype

METRICS_ABS = [
    "genome_size",
    "displacement",
    "num_voxels",
    "bone_count",
    "bone_prop",
    "fat_count",
    "fat_prop",
    "fat2_count",
    "fat2_prop",
    "phase_muscle_count",
    "phase_muscle_prop",
    "offphase_muscle_count",
    "offphase_muscle_prop",
    *BODY_METRICS,
]

METRICS_REL = [
                "uniqueness",
                "fitness",
                "age",
                "dominated_disp_nov",
                "novelty",
                "novelty_weighted"
               ]

def relative_metrics(population, args, generation, novelty_archive=None):
    uniqueness(population)
    novelty(population, novelty_archive)
    novelty_weighted(population)
    age(population, generation)
    pareto_dominance_count(population,
                           objectives=(("novelty", "max"), ("displacement", "max")), out_attr="dominated_disp_nov")
    set_fitness(population, args.fitness_metric)

def genopheno_abs_metrics(individual, args):
    genome_size(individual)
    num_voxels(individual)
    update_material_metrics(individual, args)
    update_body_metrics(individual, args)
    test_validity(individual)

def update_material_metrics(individual, args):
    if args.voxel_types == 'withbone':
        voxel_types = VOXEL_TYPES
    elif args.voxel_types == 'nobone':
        voxel_types = VOXEL_TYPES_NOBONE
    else:
        raise ValueError(f"Unsupported voxel_types: {args.voxel_types}")
    grid = np.asarray(individual.phenotype, dtype=int)
    filled_total = int((grid != 0).sum())
    individual.filled_total = filled_total
    for name, mid in voxel_types.items():
        count = int((grid == mid).sum())
        prop = (count / filled_total) if filled_total > 0 else 0.0
        setattr(individual, f"{name}_count", count)
        setattr(individual, f"{name}_prop", round(prop,2))
    muscle_aliases = {
        "muscle_h": "phase_muscle",
        "muscle_v": "offphase_muscle",
    }
    for source, alias in muscle_aliases.items():
        if source in voxel_types:
            setattr(individual, f"{alias}_count", getattr(individual, f"{source}_count"))
            setattr(individual, f"{alias}_prop", getattr(individual, f"{source}_prop"))
    if args.voxel_types == 'withbone':
        individual.fat2_count = 0
        individual.fat2_prop = 0.0

def update_body_metrics(individual, args):
    metrics = compute_body_metrics_from_phenotype(
        individual.phenotype,
        voxel_types=args.voxel_types,
        max_voxels=getattr(args, "max_voxels", None),
    )
    for metric in BODY_METRICS:
        setattr(individual, metric, metrics[metric])

def set_fitness(population, fitness_metric):
    for ind in population:
        ind.fitness = float(getattr(ind, fitness_metric, None))

def test_validity(individual):
    actuator_count = individual.phase_muscle_count + individual.offphase_muscle_count
    individual.valid = (
        actuator_count >= 2
        and individual.phase_muscle_count >= 1
        and individual.offphase_muscle_count >= 1
    )

def age(population, generation):
    for ind in population:
        age = generation - ind.born_generation + 1
        ind.age = age

def genome_size(individual):
    individual.genome_size = len(individual.genome)

def num_voxels(individual):
    individual.num_voxels = int((individual.phenotype != 0).sum())

def distance(g1, g2):
    """Exact voxel-by-voxel Hamming distance between two morphology grids."""
    a = np.asarray(g1)
    b = np.asarray(g2)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float((a != b).sum())

def uniqueness(population):
    for i, ind in enumerate(population):
        distances = []
        for j, other in enumerate(population):
            if i != j:
                d = distance(ind.phenotype, other.phenotype)
                distances.append(d / max(ind.num_voxels, other.num_voxels))
        ind.uniqueness = np.mean(distances)

def novelty_weighted(population):
    beta = 0.05
    for ind in population:
        novelty_weighted = ind.displacement * ind.novelty + beta * ind.displacement
        ind.novelty_weighted = novelty_weighted

def novelty(population, novelty_archive, k=5, M=50, embed_fn=None):
    pool = list(population) + list(novelty_archive or [])
    if embed_fn is None:
        embed_fn = lambda ind: np.array([ind.num_voxels], dtype=np.float32)
    X = np.vstack([embed_fn(ind) for ind in pool]).astype(np.float32)
    tree = KDTree(X)
    for ind in population:
        qi = embed_fn(ind).reshape(1, -1)
        _, idxs = tree.query(qi, k=min(M + 1, len(pool)))
        idxs = idxs[0]
        dists = []
        for j in idxs:
            other = pool[j]
            if other is ind:
                continue
            d = distance(ind.phenotype, other.phenotype)
            dists.append(d / max(ind.num_voxels, other.num_voxels))
        kk = min(k, len(dists))
        ind.novelty = float(np.partition(np.asarray(dists, dtype=np.float32), kk - 1)[:kk].mean()) if kk else 0.0

def pareto_dominance_count(
    population,
    objectives=(("age", "min"), ("displacement", "max")),
    out_attr="dominates_count",
):
    """
    For each individual, count how many others it Pareto-dominates
    Dominance rule:
      A dominates B iff
        - A is no worse than B in all objectives, AND
        - A is strictly better in at least one objective.
    """
    obj_specs = []
    for attr, direction in objectives:
        d = direction.strip().lower()
        obj_specs.append((attr, d))

    def dominates(a, b) -> bool:
        no_worse_all = True
        strictly_better_any = False
        for attr, d in obj_specs:
            av = getattr(a, attr)
            bv = getattr(b, attr)
            if d == "min":
                if av > bv:
                    no_worse_all = False
                    break
                if av < bv:
                    strictly_better_any = True
            else:
                if av < bv:
                    no_worse_all = False
                    break
                if av > bv:
                    strictly_better_any = True
        return no_worse_all and strictly_better_any
    for ind in population:
        setattr(ind, out_attr, 0)
    n = len(population)
    for i in range(n):
        a = population[i]
        cnt = 0
        for j in range(n):
            if i == j:
                continue
            if dominates(a, population[j]):
                cnt += 1
        setattr(a, out_attr, cnt)
