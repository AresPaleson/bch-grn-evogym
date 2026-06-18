from __future__ import annotations
from typing import Dict
import numpy as np
from algorithms.GRN import GRN

BODY_METRICS = [
    "size",
    "proportion",
    "coverage",
    "symmetry",
    "relative_number_of_joints",
    "relative_number_of_limbs",
    "total_voxel_volume",
    "bounding_box_area",
    "actuation_energy_cost",
    "environmental_contact_area",
]

def develop_body_from_genome(
    genome,
    *,
    max_voxels: int,
    cube_face_size: int,
    voxel_types: str,
    env_conditions="",
    plastic=0,
) -> np.ndarray:
    cells = GRN(
        max_voxels=max_voxels,
        cube_face_size=cube_face_size,
        genotype=genome,
        voxel_types=voxel_types,
        env_conditions=env_conditions,
        plastic=plastic,
    ).develop()
    phenotype = np.zeros(cells.shape, dtype=int)
    for idx, value in np.ndenumerate(cells):
        phenotype[idx] = value.voxel_type if value != 0 else 0
    return phenotype

def compute_body_metrics_from_phenotype(
    phenotype,
    voxel_types: str,
    max_voxels: int | None = None,
) -> Dict[str, float]:
    grid = np.asarray(phenotype, dtype=int)
    occupied = grid != 0
    occupied_count = int(occupied.sum())
    size_scale = float(max_voxels if max_voxels is not None else grid.size)
    if occupied_count == 0:
        return {
            "size": 0.0,
            "proportion": 0.0,
            "coverage": 0.0,
            "symmetry": 0.0,
            "relative_number_of_joints": 0.0,
            "relative_number_of_limbs": 0.0,
            "total_voxel_volume": 0.0,
            "bounding_box_area": 0.0,
            "actuation_energy_cost": 0.0,
            "environmental_contact_area": 0.0,
        }
    trimmed = _trim_to_occupied_bbox(grid)
    trimmed_occ = trimmed != 0
    bbox_height, bbox_width = trimmed_occ.shape
    bounding_box_area = int(bbox_width * bbox_height)
    longest_side = max(bbox_width, bbox_height)
    shortest_side = min(bbox_width, bbox_height)
    vertical_symmetry = _occupied_mirror_symmetry(trimmed_occ, axis="vertical")
    horizontal_symmetry = _occupied_mirror_symmetry(trimmed_occ, axis="horizontal")
    symmetry = max(vertical_symmetry, horizontal_symmetry)
    neighbor_count = _occupied_neighbor_count(trimmed_occ)
    limb_count = int((trimmed_occ & (neighbor_count == 1)).sum())
    actuated_count = int(np.isin(grid, [3, 4]).sum())
    max_limb_count = _max_limb_count_for_size(occupied_count)
    environmental_contact_area = int(trimmed_occ[-1, :].sum())
    return {
        "size": float(occupied_count / size_scale) if size_scale > 0 else 0.0,
        "proportion": float(shortest_side / longest_side) if longest_side > 0 else 0.0,
        "coverage": float(occupied_count / bounding_box_area) if bounding_box_area > 0 else 0.0,
        "symmetry": float(symmetry),
        "relative_number_of_joints": float(actuated_count / occupied_count),
        "relative_number_of_limbs": (
            float(min(limb_count / max_limb_count, 1.0)) if max_limb_count > 0 else 0.0
        ),
        "total_voxel_volume": float(occupied_count),
        "bounding_box_area": float(bounding_box_area),
        "actuation_energy_cost": float(actuated_count),
        "environmental_contact_area": float(environmental_contact_area),
    }

def compute_body_metrics_from_genome(
    genome,
    *,
    max_voxels: int,
    cube_face_size: int,
    voxel_types: str,
    env_conditions="",
    plastic=0,
) -> Dict[str, float]:
    phenotype = develop_body_from_genome(
        genome,
        max_voxels=max_voxels,
        cube_face_size=cube_face_size,
        voxel_types=voxel_types,
        env_conditions=env_conditions,
        plastic=plastic,
    )
    return compute_body_metrics_from_phenotype(
        phenotype,
        voxel_types=voxel_types,
        max_voxels=max_voxels,
    )

def _trim_to_occupied_bbox(grid: np.ndarray) -> np.ndarray:
    occupied = grid != 0
    rows = np.where(occupied.any(axis=1))[0]
    cols = np.where(occupied.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return grid[:0, :0]
    return grid[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

def _occupied_neighbor_count(occupied: np.ndarray) -> np.ndarray:
    neighbors = np.zeros_like(occupied, dtype=int)
    neighbors[1:, :] += occupied[:-1, :]
    neighbors[:-1, :] += occupied[1:, :]
    neighbors[:, 1:] += occupied[:, :-1]
    neighbors[:, :-1] += occupied[:, 1:]
    return neighbors

def _occupied_mirror_symmetry(occupied: np.ndarray, axis: str) -> float:
    coords = np.argwhere(occupied)
    if coords.size == 0:
        return 0.0
    height, width = occupied.shape
    mirrored = 0
    for row, col in coords:
        if axis == "vertical":
            mirror_row, mirror_col = row, width - 1 - col
        elif axis == "horizontal":
            mirror_row, mirror_col = height - 1 - row, col
        else:
            raise ValueError(f"Unsupported symmetry axis: {axis}")
        mirrored += int(occupied[mirror_row, mirror_col])
    return mirrored / len(coords)

def _max_limb_count_for_size(size: int) -> int:
    if size <= 1:
        return 0
    if size == 2:
        return 2
    return int((2 * size + 2) // 3)
