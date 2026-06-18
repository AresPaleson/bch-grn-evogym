import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT))

from evogym import get_full_connectivity

ANN_INPUT_KEYS = (
    "vel_x",
    "vel_y",
    "sin_orientation",
    "cos_orientation",
    "com_y",
    "sin_time",
    "cos_time",
    "phase_sin_time",
    "prev_action",
    "actuator_x",
    "actuator_y",
    "phase_hint",
)

def trim_phenotype_materials(phenotype):
    """
    Trim empty borders from a phenotype and return a 2D grid.
    """
    body = np.asarray(phenotype, dtype=int)
    if body.ndim != 2:
        raise ValueError(f"Expected 2D phenotype, got {body.shape}")
    x_mask = np.any(body != 0, axis=1)
    body = body[x_mask]
    if body.size == 0:
        return np.zeros((0, 0), dtype=int)
    y_mask = np.any(body != 0, axis=0)
    body = body[:, y_mask]
    return body

def _material_maps(voxel_types):
    """
    Map GRN material IDs -> EvoGym voxel IDs.
    Phase and offphase GRN muscles are mechanically identical EvoGym
    actuators; their original material IDs only drive controller phase hints.
    """
    EVOGYM = {
        "EMPTY": 0,
        "RIGID": 1,
        "SOFT": 2,
        "H_ACT": 3,
    }
    if voxel_types == "withbone":
        material_to_evogym = {
            0: EVOGYM["EMPTY"],
            1: EVOGYM["RIGID"],
            2: EVOGYM["SOFT"],
            3: EVOGYM["H_ACT"],
            4: EVOGYM["H_ACT"],
        }
    elif voxel_types == "nobone":
        material_to_evogym = {
            0: EVOGYM["EMPTY"],
            1: EVOGYM["SOFT"],
            2: EVOGYM["SOFT"],
            3: EVOGYM["H_ACT"],
            4: EVOGYM["H_ACT"],
        }
    else:
        raise ValueError(f"Unsupported voxel_types: {voxel_types}")
    return material_to_evogym

def _build_evogym_robot_data(body_materials, voxel_types):
    material_to_evogym = _material_maps(voxel_types)
    structure = np.vectorize(lambda m: material_to_evogym.get(int(m), 0), otypes=[int])(body_materials)
    structure = structure.astype(np.int32)
    connections = get_full_connectivity(structure).astype(np.int32)
    actuator_meta = np.zeros(structure.shape + (3,), dtype=np.float32)
    actuator_mask = structure == 3
    if np.any(actuator_mask):
        height, width = structure.shape
        x_coords = np.linspace(-1.0, 1.0, num=height, dtype=np.float32) if height > 1 else np.array([0.0], dtype=np.float32)
        y_coords = np.linspace(-1.0, 1.0, num=width, dtype=np.float32) if width > 1 else np.array([0.0], dtype=np.float32)
        for x_idx, y_idx in np.argwhere(actuator_mask):
            actuator_meta[x_idx, y_idx, 0] = x_coords[x_idx]
            actuator_meta[x_idx, y_idx, 1] = y_coords[y_idx]
            actuator_meta[x_idx, y_idx, 2] = 1.0 if body_materials[x_idx, y_idx] == 3 else -1.0
    return structure, connections, actuator_meta

def _derive_ann_controller(individual, args):
    input_size = len(ANN_INPUT_KEYS)
    hidden_size = int(getattr(args, "evogym_ann_hidden_size", 8))
    action_bias = float(getattr(args, "evogym_action_bias", 1.0))
    action_amplitude = float(getattr(args, "evogym_action_amplitude", 0.6))
    sine_period = float(getattr(args, "evogym_sine_period", 40.0))
    sine_mix = float(getattr(args, "evogym_sine_mix", 0.35))
    param_count = (input_size * hidden_size) + hidden_size + hidden_size + 1
    source = np.asarray(individual.genome, dtype=np.float32).flatten()
    if source.size == 0:
        source = np.zeros(1, dtype=np.float32)
    sample_points = np.linspace(0, source.size - 1, num=param_count, dtype=np.float32)
    sampled = np.interp(sample_points, np.arange(source.size, dtype=np.float32), source)
    params = ((sampled * 2.0) - 1.0).astype(np.float32)
    return {
        "controller_type": "ann",
        "input_keys": list(ANN_INPUT_KEYS),
        "hidden_size": hidden_size,
        "action_bias": action_bias,
        "action_amplitude": action_amplitude,
        "sine_period": max(sine_period, 1.0),
        "sine_mix": float(np.clip(sine_mix, 0.0, 1.0)),
        "weights": params.tolist(),
    }

def prepare_robot_files(individual, args):
    """
    Prepare EvoGym robot artifacts from an evolved phenotype.
    Keeps the old function name so the EA loop can call it unchanged.
    """
    body = trim_phenotype_materials(individual.phenotype)
    if body.size == 0:
        raise ValueError(f"Robot {individual.id} has an empty phenotype and cannot be prepared for EvoGym.")
    structure, connections, actuator_meta = _build_evogym_robot_data(
        body, args.voxel_types
    )
    individual.evogym_structure = structure
    individual.evogym_connections = connections
    individual.evogym_actuator_meta = actuator_meta
    individual.evogym_controller = _derive_ann_controller(individual, args)
