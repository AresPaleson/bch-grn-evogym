import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_EVOGYM_ENV_NAME = "DownStepper-v0"

FLAT_CEILING_ENV_NAME = "FlatCeiling-v0"

GENERATED_WORLD_DIR = "_generated_worlds"

DEFAULT_LEFT_WALL_HEIGHT = 8

EVOGYM_DEFAULT_USTATIC = 0.5

EVOGYM_DEFAULT_UDYNAMIC = 0.2

FLAT_CEILING_GAP_BLOCKS = 6

FLAT_CEILING_PIT_SIZE_BLOCKS = 6

FLAT_CEILING_SHALLOW_PITS = (
    (30, 1),
    (60, 2),
)

def _grid_to_json_object(occupied_cells, width: int):
    indices = sorted((y * width) + x for x, y in occupied_cells)
    neighbors = {}
    occupied_set = set(occupied_cells)
    for x, y in occupied_cells:
        idx = (y * width) + x
        linked = []
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (nx, ny) in occupied_set:
                linked.append((ny * width) + nx)
        neighbors[str(idx)] = sorted(linked)
    return {
        "indices": indices,
        "types": [5] * len(indices),
        "neighbors": neighbors,
    }

def _is_connected_grid_object(occupied_cells) -> bool:
    occupied = set(occupied_cells)
    if not occupied:
        return False
    stack = [next(iter(occupied))]
    seen = {stack[0]}
    while stack:
        x, y = stack.pop()
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbor in occupied and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(occupied)

def _generated_world_dir(args) -> Path:
    out_root = Path(getattr(args, "out_path", "tmp_out"))
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    generated_dir = out_root / GENERATED_WORLD_DIR
    generated_dir.mkdir(parents=True, exist_ok=True)
    return generated_dir

def _left_wall_cells_for_world(occupied_cells, left_wall_height: int):
    occupied = set(occupied_cells)
    if not occupied:
        return set()
    wall_height = max(1, int(left_wall_height))
    min_x = min(x for x, _ in occupied)
    anchor_rows = sorted(y for x, y in occupied if x == min_x)
    if not anchor_rows:
        return set()
    anchor_y = anchor_rows[0]
    downward_wall = {(min_x, y) for y in range(anchor_y, anchor_y + wall_height)}
    if _is_connected_grid_object(occupied | downward_wall):
        return downward_wall
    upward_wall = {(min_x, y) for y in range(anchor_y - wall_height + 1, anchor_y + 1)}
    return upward_wall

def _add_left_wall_to_world(base_world: dict, left_wall_height: int) -> dict:
    width = int(base_world["grid_width"])
    ground = base_world["objects"]["ground"]
    occupied = {
        (idx % width, idx // width)
        for idx in ground["indices"]
    }
    occupied.update(_left_wall_cells_for_world(occupied, left_wall_height))
    min_y = min(y for _, y in occupied)
    if min_y < 0:
        occupied = {(x, y - min_y) for x, y in occupied}
    height = max(int(base_world["grid_height"]), max(y for _, y in occupied) + 1)
    world = dict(base_world)
    world["grid_height"] = height
    world["objects"] = dict(base_world["objects"])
    world["objects"]["ground"] = _grid_to_json_object(occupied, width)
    return world

def _make_flat_ceiling_world(width: int, gap_blocks: int) -> dict:
    gap = max(1, int(gap_blocks))
    pit_size = FLAT_CEILING_PIT_SIZE_BLOCKS
    platform_y = pit_size
    ceiling_y = platform_y + gap + 1
    pit_inner_start = width - pit_size - 1
    pit_inner_end = width - 2
    shallow_pit_columns = {
        x
        for start_x, pit_width in FLAT_CEILING_SHALLOW_PITS
        for x in range(start_x, start_x + pit_width)
        if 0 < x < pit_inner_start
    }
    ground_segments = []
    segment = set()
    for x in range(0, pit_inner_start):
        if x in shallow_pit_columns:
            if segment:
                ground_segments.append(segment)
                segment = set()
            continue
        segment.add((x, platform_y))
    if segment:
        ground_segments.append(segment)
    ceiling_cells = {(x, ceiling_y) for x in range(width)}
    left_bar_cells = {(0, y) for y in range(platform_y + 1, ceiling_y)}
    shallow_pit_floor_objects = {
        f"shallow_pit_floor_x{start_x}": {
            (x, platform_y - 1)
            for x in range(start_x, start_x + pit_width)
            if x in shallow_pit_columns
        }
        for start_x, pit_width in FLAT_CEILING_SHALLOW_PITS
    }
    pit_floor_cells = {(x, 0) for x in range(pit_inner_start, pit_inner_end + 1)}
    pit_left_wall_cells = {(pit_inner_start - 1, y) for y in range(0, platform_y)}
    right_bar_cells = {(width - 1, y) for y in range(0, ceiling_y)}
    objects = {
        "ceiling": _grid_to_json_object(ceiling_cells, width),
        "left_bar": _grid_to_json_object(left_bar_cells, width),
        "pit_floor": _grid_to_json_object(pit_floor_cells, width),
        "pit_left_wall": _grid_to_json_object(pit_left_wall_cells, width),
        "right_bar": _grid_to_json_object(right_bar_cells, width),
    }
    objects.update(
        {
            "ground" if index == 0 else f"ground_{index}": _grid_to_json_object(cells, width)
            for index, cells in enumerate(ground_segments)
        }
    )
    objects.update(
        {
            name: _grid_to_json_object(cells, width)
            for name, cells in shallow_pit_floor_objects.items()
            if cells
        }
    )
    return {
        "grid_width": width,
        "grid_height": ceiling_y + 1,
        "objects": objects,
    }

def _ensure_generated_flat_ceiling_world(args) -> Path:
    width = int(getattr(args, "evogym_flat_ceiling_width", 100))
    gap_blocks = int(getattr(args, "evogym_flat_ceiling_gap_blocks", FLAT_CEILING_GAP_BLOCKS))
    width = max(FLAT_CEILING_PIT_SIZE_BLOCKS + 4, width)
    generated_dir = _generated_world_dir(args)
    generated_path = generated_dir / f"{FLAT_CEILING_ENV_NAME}-w{width}-gap{gap_blocks}-pit6.json"
    generated_path.write_text(json.dumps(_make_flat_ceiling_world(width, gap_blocks), indent=4))
    return generated_path

def _ensure_generated_named_world(args, env_name: str, base_path: Path) -> Path:
    if env_name == FLAT_CEILING_ENV_NAME:
        return _ensure_generated_flat_ceiling_world(args)
    if not bool(int(getattr(args, "evogym_left_wall", 0))):
        return base_path
    wall_height = int(getattr(args, "evogym_left_wall_height", DEFAULT_LEFT_WALL_HEIGHT))
    generated_dir = _generated_world_dir(args)
    generated_path = generated_dir / f"{env_name}-left-wall-h{wall_height}.json"
    base_world = json.loads(base_path.read_text())
    base_width = int(base_world["grid_width"])
    base_ground = base_world["objects"]["ground"]
    base_occupied = {
        (idx % base_width, idx // base_width)
        for idx in base_ground["indices"]
    }
    wall_occupied = _left_wall_cells_for_world(base_occupied, wall_height)
    if not _is_connected_grid_object(base_occupied | wall_occupied):
        print(
            f"[WORLD] Skipping left wall for {env_name}: "
            "it would make the terrain object disconnected."
        )
        return base_path
    custom_world = _add_left_wall_to_world(base_world, wall_height)
    generated_path.write_text(json.dumps(custom_world, indent=4))
    return generated_path

def _ann_forward(inputs: np.ndarray, params: np.ndarray, hidden_size: int) -> np.ndarray:
    input_size = inputs.shape[1]
    split_1 = input_size * hidden_size
    split_2 = split_1 + hidden_size
    split_3 = split_2 + hidden_size
    w1 = params[:split_1].reshape(input_size, hidden_size)
    b1 = params[split_1:split_2]
    w2 = params[split_2:split_3].reshape(hidden_size, 1)
    b2 = params[split_3]
    hidden = np.tanh(inputs @ w1 + b1)
    outputs = np.tanh(hidden @ w2 + b2)
    return outputs.reshape(-1)

def _resolve_steps(args) -> int:
    steps = int(getattr(args, "evogym_steps", 500))
    return max(1, steps)

def _resolve_workers(args, n_jobs: int) -> int:
    if int(getattr(args, "evogym_headless", 1)) == 0:
        return 1
    requested = int(getattr(args, "evogym_num_workers", 0))
    if requested > 0:
        return max(1, min(requested, n_jobs))
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, n_jobs))

def _should_isolate_tasks(args) -> bool:
    """
    On Windows, long EvoGym runs are more stable if each robot is evaluated in
    a fresh spawned subprocess, so native simulator state cannot accumulate in
    one long-lived worker.
    """
    explicit = getattr(args, "evogym_isolate_tasks", None)
    if explicit is not None:
        return bool(int(explicit))
    return os.name == "nt" and int(getattr(args, "evogym_headless", 1)) == 1

def _resolve_world_path(args) -> str:
    world_path = getattr(args, "evogym_world_path", None)
    env_name = str(getattr(args, "evogym_env_name", "") or DEFAULT_EVOGYM_ENV_NAME).strip()
    if world_path is None:
        base_path = ROOT / "evogym" / "evogym" / "envs" / "sim_files" / f"{env_name}.json"
        if env_name == FLAT_CEILING_ENV_NAME:
            world_path = _ensure_generated_named_world(args, env_name, base_path)
        elif not base_path.exists():
            raise FileNotFoundError(f"Could not find EvoGym world for evogym_env_name={env_name!r}: {base_path}")
        else:
            world_path = _ensure_generated_named_world(args, env_name, base_path)
    world_path = Path(world_path)
    if not world_path.is_absolute():
        world_path = ROOT / world_path
    return str(world_path)

def _resolve_displacement_axis(args) -> int:
    axis = getattr(args, "evogym_displacement_axis", None)
    if axis is not None:
        return int(axis)
    return 0

def _apply_friction(sim, ustatic: float, udynamic: float) -> None:
    try:
        sim.set_friction(float(ustatic), float(udynamic))
        return
    except AttributeError:
        pass
    is_default = (
        np.isclose(float(ustatic), EVOGYM_DEFAULT_USTATIC)
        and np.isclose(float(udynamic), EVOGYM_DEFAULT_UDYNAMIC)
    )
    if not is_default:
        raise RuntimeError(
            "This EvoGym simulator binary does not expose set_friction. "
            "Rebuild evogym/simulator_cpp after the C++ binding changes, "
            "or use the compiled defaults ustatic=0.5 and udynamic=0.2."
        )

def _simulate_one_robot(task: Dict) -> Tuple[int, float, str]:
    """Returns (robot_id, task displacement, error_msg)."""
    from evogym import EvoWorld, EvoSim
    robot_id = int(task["id"])
    structure = task["structure"]
    connections = task["connections"]
    actuator_meta = task["actuator_meta"]
    world_path = task["world_path"]
    bias = float(task["action_bias"])
    amplitude = float(task["action_amplitude"])
    sine_period = max(float(task.get("sine_period", 40.0)), 1.0)
    sine_mix = float(np.clip(task.get("sine_mix", 0.35), 0.0, 1.0))
    controller_type = str(task["controller_type"])
    hidden_size = int(task["hidden_size"])
    ann_weights = np.asarray(task["weights"], dtype=np.float32)
    sim_steps = int(task["sim_steps"])
    init_x = int(task["init_x"])
    init_y = int(task["init_y"])
    headless = bool(int(task["headless"]))
    render_mode = str(task["render_mode"])
    ustatic = float(task["ustatic"])
    udynamic = float(task["udynamic"])
    displacement_axis = int(task.get("displacement_axis", 0))
    record_video = bool(int(task.get("record_video", 0)))
    record_video_path = str(task.get("record_video_path", "")).strip()
    record_video_fps = max(1, int(task.get("record_video_fps", 50)))
    record_video_stride = max(1, int(task.get("record_video_stride", 1)))
    try:
        world = EvoWorld.from_json(world_path)
        world.add_from_array(
            name="robot",
            structure=structure,
            x=init_x,
            y=init_y,
            connections=connections,
        )
        sim = EvoSim(world)
        _apply_friction(sim, ustatic, udynamic)
        sim.reset()
        viewer = None
        frames = []
        if not headless or record_video:
            from evogym.viewer import EvoViewer
            viewer = EvoViewer(sim)
            viewer.track_objects("robot")
        actuator_indices = sim.get_actuator_indices("robot").astype(int).flatten()
        actuator_meta_flat = np.flipud(actuator_meta).reshape(-1, actuator_meta.shape[-1])
        actuator_features = (
            actuator_meta_flat[actuator_indices] if actuator_indices.size else np.zeros((0, 3), dtype=np.float32)
        )
        prev_action = np.full(actuator_indices.size, bias, dtype=np.float32)
        p0 = sim.object_pos_at_time(sim.get_time(), "robot")
        start = float(np.mean(p0[displacement_axis]))
        for t in range(sim_steps):
            if actuator_indices.size:
                pos = sim.object_pos_at_time(sim.get_time(), "robot")
                vel = sim.object_vel_at_time(sim.get_time(), "robot")
                orientation = float(sim.object_orientation_at_time(sim.get_time(), "robot"))
                com_pos = np.mean(pos, axis=1)
                com_vel = np.mean(vel, axis=1)
                base_phase = (2.0 * np.pi * t) / sine_period
                sin_time = np.sin(base_phase)
                cos_time = np.cos(base_phase)
                phase_wave = sin_time * actuator_features[:, 2]
                if controller_type != "ann":
                    raise ValueError(f"Unsupported controller_type: {controller_type}")
                ann_inputs = np.column_stack(
                    (
                        np.full(actuator_indices.size, np.clip(com_vel[0], -5.0, 5.0), dtype=np.float32),
                        np.full(actuator_indices.size, np.clip(com_vel[1], -5.0, 5.0), dtype=np.float32),
                        np.full(actuator_indices.size, np.sin(orientation), dtype=np.float32),
                        np.full(actuator_indices.size, np.cos(orientation), dtype=np.float32),
                        np.full(actuator_indices.size, np.clip(com_pos[1], 0.0, 10.0), dtype=np.float32),
                        np.full(actuator_indices.size, sin_time, dtype=np.float32),
                        np.full(actuator_indices.size, cos_time, dtype=np.float32),
                        phase_wave.astype(np.float32),
                        np.clip((prev_action - bias) / max(amplitude, 1e-6), -1.0, 1.0),
                        actuator_features[:, 0],
                        actuator_features[:, 1],
                        actuator_features[:, 2],
                    )
                )
                ann_output = _ann_forward(ann_inputs, ann_weights, hidden_size)
                regularized_output = ((1.0 - sine_mix) * ann_output) + (sine_mix * phase_wave)
                regularized_output = np.clip(regularized_output, -1.0, 1.0)
                action = bias + amplitude * regularized_output
                action = np.clip(action, 0.6, 1.6).astype(np.float64)
                prev_action = action.astype(np.float32)
                sim.set_action("robot", action)
            unstable = sim.step()
            if viewer is not None:
                if not headless:
                    viewer.render(render_mode)
                if record_video and t % record_video_stride == 0:
                    frames.append(np.asarray(viewer.render("img"), dtype=np.uint8))
            if unstable:
                break
        p1 = sim.object_pos_at_time(sim.get_time(), "robot")
        finish = float(np.mean(p1[displacement_axis]))
        displacement = finish - start
        if viewer is not None:
            viewer.close()
        if record_video and record_video_path and frames:
            import imageio
            output_path = Path(record_video_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(output_path, frames, fps=record_video_fps)
            print(f"[VIDEO] saved {len(frames)} frames to: {output_path}")
        return robot_id, displacement, ""
    except Exception as exc:
        return robot_id, float("-inf"), f"{type(exc).__name__}: {exc}"

def simulate_evogym_batch(population, args):
    """
    Evaluate all valid individuals in EvoGym and write displacement into each individual.
    """
    sim_steps = _resolve_steps(args)
    init_x = int(getattr(args, "evogym_init_x", 3))
    init_y = int(getattr(args, "evogym_init_y", FLAT_CEILING_PIT_SIZE_BLOCKS + 1))
    default_bias = float(getattr(args, "evogym_action_bias", 1.0))
    default_amplitude = float(getattr(args, "evogym_action_amplitude", 0.6))
    ustatic = float(getattr(args, "ustatic", EVOGYM_DEFAULT_USTATIC))
    udynamic = float(getattr(args, "udynamic", EVOGYM_DEFAULT_UDYNAMIC))
    headless = int(getattr(args, "evogym_headless", 1))
    render_mode = str(getattr(args, "evogym_render_mode", "screen"))
    record_video = int(getattr(args, "evogym_record_video", 0))
    record_video_path = str(getattr(args, "evogym_record_video_path", ""))
    record_video_fps = int(getattr(args, "evogym_record_video_fps", 50))
    record_video_stride = int(getattr(args, "evogym_record_video_stride", 1))
    world_path = _resolve_world_path(args)
    displacement_axis = _resolve_displacement_axis(args)
    id_to_ind = {ind.id: ind for ind in population}
    tasks: List[Dict] = []
    for ind in population:
        if not getattr(ind, "valid", True):
            continue
        if not hasattr(ind, "evogym_structure"):
            raise RuntimeError(
                f"Robot {ind.id} missing EvoGym payload. "
                "Call prepare_robot_files(individual, args) before simulation."
            )
        ctrl = getattr(ind, "evogym_controller", {})
        task = {
            "id": ind.id,
            "structure": ind.evogym_structure,
            "connections": ind.evogym_connections,
            "actuator_meta": ind.evogym_actuator_meta,
            "controller_type": ctrl.get("controller_type", "ann"),
            "hidden_size": ctrl.get("hidden_size", 8),
            "weights": ctrl.get("weights", []),
            "action_bias": ctrl.get("action_bias", default_bias),
            "action_amplitude": ctrl.get("action_amplitude", default_amplitude),
            "sine_period": ctrl.get("sine_period", float(getattr(args, "evogym_sine_period", 40.0))),
            "sine_mix": ctrl.get("sine_mix", float(getattr(args, "evogym_sine_mix", 0.35))),
            "sim_steps": sim_steps,
            "init_x": init_x,
            "init_y": init_y,
            "headless": headless,
            "render_mode": render_mode,
            "record_video": record_video,
            "record_video_path": record_video_path,
            "record_video_fps": record_video_fps,
            "record_video_stride": record_video_stride,
            "world_path": world_path,
            "displacement_axis": displacement_axis,
            "ustatic": ustatic,
            "udynamic": udynamic,
        }
        tasks.append(task)
    if not tasks:
        print("[SIM-DONE] total=0 ok=0 failed=0")
        return
    n_workers = _resolve_workers(args, len(tasks))
    isolate_tasks = _should_isolate_tasks(args)
    ok = 0
    failed = 0
    if n_workers == 1 and not isolate_tasks:
        for task in tasks:
            rid, disp, err = _simulate_one_robot(task)
            ind = id_to_ind[rid]
            ind.displacement = float(disp)
            if err:
                failed += 1
                print(f"[SIM-FAIL] {rid}: {err}")
            else:
                ok += 1
    elif isolate_tasks:
        ctx = mp.get_context("spawn")
        try:
            with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
                results = [pool.apply_async(_simulate_one_robot, (task,)) for task in tasks]
                for result in results:
                    rid, disp, err = result.get()
                    ind = id_to_ind[rid]
                    ind.displacement = float(disp)
                    if err:
                        failed += 1
                        print(f"[SIM-FAIL] {rid}: {err}")
                    else:
                        ok += 1
        except Exception as exc:
            print(
                "[SIM-WARN] Isolated EvoGym workers failed "
                f"({type(exc).__name__}: {exc}). Retrying this batch in-process."
            )
            ok = 0
            failed = 0
            n_workers = 1
            for task in tasks:
                rid, disp, err = _simulate_one_robot(task)
                ind = id_to_ind[rid]
                ind.displacement = float(disp)
                if err:
                    failed += 1
                    print(f"[SIM-FAIL] {rid}: {err}")
                else:
                    ok += 1
    else:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = [ex.submit(_simulate_one_robot, t) for t in tasks]
                for fut in as_completed(futs):
                    rid, disp, err = fut.result()
                    ind = id_to_ind[rid]
                    ind.displacement = float(disp)
                    if err:
                        failed += 1
                        print(f"[SIM-FAIL] {rid}: {err}")
                    else:
                        ok += 1
        except BrokenProcessPool:
            print(
                "[SIM-WARN] EvoGym worker process exited unexpectedly. "
                "Retrying this batch in a single process."
            )
            ok = 0
            failed = 0
            n_workers = 1
            for task in tasks:
                rid, disp, err = _simulate_one_robot(task)
                ind = id_to_ind[rid]
                ind.displacement = float(disp)
                if err:
                    failed += 1
                    print(f"[SIM-FAIL] {rid}: {err}")
                else:
                    ok += 1
    print(
        f"[SIM-DONE] total={len(tasks)} ok={ok} failed={failed} "
        f"workers={n_workers} isolated={int(isolate_tasks)} steps={sim_steps}"
    )
