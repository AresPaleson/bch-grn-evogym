import os
import sys
from pathlib import Path
from glob import glob
from itertools import zip_longest
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.append(str(ROOT))

from algorithms.EA_classes import Robot, GenerationSurvivor
from algorithms.voxel_types import (
    VOXEL_TYPES,
    VOXEL_TYPES_COLORS,
    VOXEL_TYPES_NOBONE,
    VOXEL_TYPES_COLORS_NOBONE,
)

from utils.draw import draw_phenotype
from utils.config import Config
from utils.body_metrics import develop_body_from_genome

try:
    from PIL import Image, ImageDraw, ImageFont, Image
    PIL_AVAILABLE = True

except Exception:
    PIL_AVAILABLE = False

def newest_file_in(dirpath):
    files = [p for p in glob(os.path.join(dirpath, "*.*")) if os.path.isfile(p)]
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def get_voxel_rendering(voxel_types_name):
    if voxel_types_name == "nobone":
        return VOXEL_TYPES_NOBONE, VOXEL_TYPES_COLORS_NOBONE
    return VOXEL_TYPES, VOXEL_TYPES_COLORS

def fetch_best_robot_of_gen(session, gen):
    """Return (Robot, GenerationSurvivor) for highest-fitness survivor of generation 'gen'."""
    row = (
        session.query(Robot, GenerationSurvivor)
        .join(GenerationSurvivor, GenerationSurvivor.robot_id == Robot.robot_id)
        .filter(GenerationSurvivor.generation == gen)
        .order_by(GenerationSurvivor.fitness.desc().nullslast())
        .first()
    )
    return row

def fetch_robot(session, rid):
    if rid is None:
        return None
    return session.query(Robot).filter(Robot.robot_id == rid).first()

def best_survivor_row(session, rid):
    """Best (highest-fitness) GenerationSurvivor row for robot_id (may be from any gen)."""
    if rid is None:
        return None
    return (
        session.query(GenerationSurvivor)
        .filter(GenerationSurvivor.robot_id == rid)
        .order_by(GenerationSurvivor.fitness.desc().nullslast())
        .first()
    )


def genome_distance(child_genome, parent_genome):
    pairs = zip_longest(child_genome or [], parent_genome or [], fillvalue=0.0)
    diffs = [abs(float(child) - float(parent)) for child, parent in pairs]
    return sum(diffs) / len(diffs) if diffs else 0.0


def _dist_safe(g_child, g_parent):
    try:
        return genome_distance(g_child, g_parent)
    except Exception:
        return float("inf")

def choose_most_similar_parent(session, robot):
    """
    Choose the parent with the smaller genome distance. Returns:
      (chosen_parent_robot, chosen_parent_surv_row, dist_info_dict)
    dist_info_dict = {
        "d_p1": float or None,
        "d_p2": float or None,
        "picked": "P1" | "P2" | None
    }
    """
    p1 = fetch_robot(session, getattr(robot, "parent1_id", None))
    p2 = fetch_robot(session, getattr(robot, "parent2_id", None))
    if p1 is None and p2 is None:
        return None, None, {"d_p1": None, "d_p2": None, "picked": None}
    d1 = _dist_safe(robot.genome, p1.genome) if p1 is not None else None
    d2 = _dist_safe(robot.genome, p2.genome) if p2 is not None else None
    picked = None
    chosen_parent = None
    if p1 is not None and p2 is not None:
        if d1 is None and d2 is not None:
            chosen_parent, picked = p2, "P2"
        elif d2 is None and d1 is not None:
            chosen_parent, picked = p1, "P1"
        else:
            chosen_parent, picked = (p1, "P1") if (d1 <= d2) else (p2, "P2")
    elif p1 is not None:
        chosen_parent, picked = p1, "P1"
    else:
        chosen_parent, picked = p2, "P2"
    parent_surv = best_survivor_row(session, getattr(chosen_parent, "robot_id", None))
    return chosen_parent, parent_surv, {"d_p1": d1, "d_p2": d2, "picked": picked}

def build_lineage(session, leaf_robot, max_back_gens):
    """
    Make a single-path lineage by repeatedly choosing the most-similar parent.
    Returns a list (oldest -> youngest) of dicts with:
      {
        "robot": Robot,
        "surv": GenerationSurvivor or None,
        "dist_to_parents": {"d_p1": float|None, "d_p2": float|None, "picked": "P1"|"P2"|None}
      }
    The first (oldest) node has dist_to_parents = None since it has no parent in the chain.
    """
    chain = []
    leaf_surv = best_survivor_row(session, leaf_robot.robot_id)
    chain.append({
        "robot": leaf_robot,
        "surv": leaf_surv,
        "dist_to_parents": None,
    })
    steps = 0
    current = leaf_robot
    while steps < max_back_gens:
        parent, parent_surv, _ = choose_most_similar_parent(session, current)
        if parent is None:
            break
        chain.append({
            "robot": parent,
            "surv": parent_surv,
            "dist_to_parents": None,
        })
        current = parent
        steps += 1
    chain.reverse()
    for i in range(1, len(chain)):
        child = chain[i]["robot"]
        parent = chain[i-1]["robot"]
        parent1 = fetch_robot(session, getattr(child, "parent1_id", None))
        parent2 = fetch_robot(session, getattr(child, "parent2_id", None))
        d_p1 = _dist_safe(child.genome, parent1.genome) if parent1 is not None else None
        d_p2 = _dist_safe(child.genome, parent2.genome) if parent2 is not None else None
        picked = (
            "P1"
            if getattr(child, "parent1_id", None) == getattr(parent, "robot_id", None)
            else "P2"
            if getattr(child, "parent2_id", None) == getattr(parent, "robot_id", None)
            else None
        )
        chain[i]["dist_to_parents"] = {"d_p1": d_p1, "d_p2": d_p2, "picked": picked}
    return chain

def draw_lineage_nodes(
    out_dir,
    lineage,
    *,
    max_voxels,
    cube_face_size,
    voxel_types_name,
    env_conditions,
    plastic,
):
    """
    For each lineage entry, draw phenotype and save deterministic filename (preserving extension).
    Returns: list of (image_path, robot_id, fitness, dist_info) in oldest->youngest order.
    dist_info is None for the first (root) node; otherwise a dict with d_p1, d_p2, picked.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    voxel_types, voxel_types_colors = get_voxel_rendering(voxel_types_name)
    for idx, entry in enumerate(lineage):
        robot = entry["robot"]
        surv = entry["surv"]
        pre_existing = set(glob(os.path.join(out_dir, "*.*")))
        phenotype = develop_body_from_genome(
            robot.genome,
            max_voxels=max_voxels,
            cube_face_size=cube_face_size,
            voxel_types=voxel_types_name,
            env_conditions=env_conditions,
            plastic=plastic,
        )
        fitness = round(surv.fitness, 4) if (surv and surv.fitness is not None) else None
        draw_phenotype(
            phenotype,
            robot.robot_id,
            cube_face_size,
            idx,
            fitness,
            out_dir,
            voxel_types,
            voxel_types_colors,
        )
        post_existing = set(glob(os.path.join(out_dir, "*.*")))
        new_files = list(post_existing - pre_existing)
        if not new_files:
            newest = newest_file_in(out_dir)
        else:
            new_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            newest = new_files[0]
        if newest is None or not os.path.exists(newest):
            raise RuntimeError("Could not locate phenotype image produced by draw_phenotype.")
        _, ext = os.path.splitext(newest)
        if not ext:
            ext = ".png"
        new_name = os.path.join(out_dir, f"{idx:02d}_robot_{robot.robot_id}{ext}")
        if os.path.abspath(newest) != os.path.abspath(new_name):
            os.replace(newest, new_name)
        saved.append((new_name, robot.robot_id, fitness, entry["dist_to_parents"]))
    return saved

def compose_vertical_tree(out_dir, nodes, margin=32, v_gap=28, arrow_len=20, remove_intermediate=True):
    """
    Compose a single PNG: oldest at top, youngest at bottom.
    `nodes` is a list of (image_path, robot_id, fitness, dist_info).
    dist_info is None for the oldest; otherwise contains {"d_p1", "d_p2", "picked"}.
    """
    if not PIL_AVAILABLE:
        print("Pillow not available; skipping tree composition. Install with `pip install pillow`.")
        return None
    from PIL import Image, ImageDraw, ImageFont
    FONT_SIZE = 100
    BOTTOM_EXTRA = FONT_SIZE * 2

    def _find_font():
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "arial.ttf",
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, FONT_SIZE), p
                except Exception:
                    pass
        return None, None

    def _render_text_big(draw_ctx, text):
        tiny_font = ImageFont.load_default()
        bbox = draw_ctx.textbbox((0, 0), text, font=tiny_font)
        w, h = max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((0, 0), text, fill=(0, 0, 0, 255), font=tiny_font)
        scale = max(2, int(round(FONT_SIZE / float(h))))
        big = img.resize((w * scale, h * scale), resample=Image.NEAREST)
        return big

    def measure_text(draw_ctx, text, font):
        if font is not None:
            l, t, r, b = draw_ctx.textbbox((0, 0), text, font=font)
            return r - l, b - t
        else:
            tmp = _render_text_big(draw_ctx, text)
            return tmp.width, tmp.height

    def draw_caption(canvas, draw_ctx, center_x, y, text, font):
        if font is not None:
            l, t, r, b = draw_ctx.textbbox((0, 0), text, font=font)
            w, h = r - l, b - t
            x = center_x - w // 2
            draw_ctx.text((x, y), text, fill=(0, 0, 0), font=font)
            return h
        else:
            img = _render_text_big(draw_ctx, text)
            x = center_x - img.width // 2
            canvas.alpha_composite(img, (x, y))
            return img.height
    font, font_path = _find_font()
    if font_path:
        print(f"[family_tree] Using font: {font_path} @ {FONT_SIZE}px")
    else:
        print("[family_tree] No TTF font found; using scaled bitmap fallback.")
    imgs = [Image.open(p).convert("RGBA") for (p, _, _, _) in nodes]
    max_w = max(im.width for im in imgs)
    captions = []
    for _, rid, fit, dist in nodes:
        base = f"ID: {rid} | Fitness: {fit if fit is not None else 'NA'}"
        if dist is not None:
            d1 = "NA" if dist["d_p1"] is None or dist["d_p1"] == float('inf') else f"{dist['d_p1']:.3f}"
            d2 = "NA" if dist["d_p2"] is None or dist["d_p2"] == float('inf') else f"{dist['d_p2']:.3f}"
            picked = dist["picked"] if dist["picked"] else "NA"
            base += f" | dP1={d1}, dP2={d2}, picked={picked}"
        captions.append(base)
    dummy = Image.new("RGBA", (1, 1))
    ddraw = ImageDraw.Draw(dummy)
    cap_sizes = [measure_text(ddraw, c, font) for c in captions]
    cap_heights = [h + 8 for (_, h) in cap_sizes]
    total_h = margin
    for im, ch in zip(imgs, cap_heights):
        total_h += im.height + ch + v_gap
    total_h += BOTTOM_EXTRA
    canvas_w = max_w + margin * 2
    canvas_h = total_h
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    cx = canvas_w // 2
    y = margin
    for i, im in enumerate(imgs):
        x = cx - im.width // 2
        canvas.paste(im, (x, y), im)
        cap_y = y + im.height + 4
        h = draw_caption(canvas, draw, cx, cap_y, captions[i], font)
        y_next_start = cap_y + h
        if i < len(imgs) - 1:
            y_line_start = y_next_start + 6
            y_line_end = y_line_start + arrow_len
            draw.line([(cx, y_line_start), (cx, y_line_end)], fill=(0, 0, 0), width=2)
            draw.polygon(
                [(cx - 5, y_line_end - 2), (cx + 5, y_line_end - 2), (cx, y_line_end + 6)],
                fill=(0, 0, 0),
            )
        y = y_next_start + v_gap + arrow_len
    out_path = os.path.join(out_dir, "family_tree.png")
    canvas.save(out_path)
    if remove_intermediate:
        for p, _, _, _ in nodes:
            try:
                os.remove(p)
            except Exception:
                pass
    return out_path

def main():
    args = Config()._get_params()
    experiments = args.experiments.split(",")
    runs = list(map(int, args.runs.split(","))) if args.runs else [args.run]
    voxel_types_list = args.voxel_types.split(",") if args.voxel_types else ["withbone"]
    max_back = int(getattr(args, "x_gens", 5) or 5)
    for exp_idx, experiment_name in enumerate(experiments):
        print(experiment_name)
        voxel_types_for_exp = (
            voxel_types_list[exp_idx]
            if exp_idx < len(voxel_types_list)
            else voxel_types_list[-1]
        )
        for run in runs:
            print(" run:", run)
            out_dir = f"{args.out_path}/{args.study_name}/analysis/family_trees/{experiment_name}/run_{run}"
            os.makedirs(out_dir, exist_ok=True)
            db_path = f"{args.out_path}/{args.study_name}/{experiment_name}/run_{run}/run_{run}"
            if not os.path.exists(db_path):
                raise FileNotFoundError(
                    f"Database not found at '{db_path}'. "
                    "Make sure this matches the Experiment's DB location."
                )
            engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
            Session = sessionmaker(bind=engine, expire_on_commit=False)
            with Session() as session:
                total_survivor_gens = session.query(func.count(GenerationSurvivor.generation.distinct())).scalar()
                if total_survivor_gens == 0:
                    print("  (no completed generations found in DB)")
                    continue
                final_gen = session.query(func.max(GenerationSurvivor.generation)).scalar()
                if final_gen is None:
                    print("  (could not determine final generation)")
                    continue
                best_row = fetch_best_robot_of_gen(session, final_gen)
                if not best_row:
                    print(f"  (no survivors for final gen {final_gen})")
                    continue
                leaf_robot, leaf_surv = best_row
                print(f"  Final-gen best robot: {leaf_robot.robot_id} (fitness={leaf_surv.fitness})")
                lineage = build_lineage(session, leaf_robot, max_back_gens=max_back)
                print(f"  Lineage length (oldest->youngest): {len(lineage)}")
                nodes = draw_lineage_nodes(
                    out_dir=out_dir,
                    lineage=lineage,
                    max_voxels=args.max_voxels,
                    cube_face_size=args.cube_face_size,
                    voxel_types_name=voxel_types_for_exp,
                    env_conditions=args.env_conditions,
                    plastic=args.plastic,
                )
                tree_path = compose_vertical_tree(out_dir, nodes, remove_intermediate=True)
                if tree_path:
                    print(f"  Family tree saved: {tree_path}")
                else:
                    print(f"  Node images saved in {out_dir} (install Pillow to compose a tree PNG).")

if __name__ == "__main__":
    main()
