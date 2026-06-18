import argparse

EVOGYM_ENVIRONMENT_OPTIONS = (
    "FlatCeiling-v0",
    "DownStepper-v0",
)

EVOGYM_ENVIRONMENT_ALIASES = {
    "flatceiling": "FlatCeiling-v0",
    "flatceilingv0": "FlatCeiling-v0",
    "ceiling": "FlatCeiling-v0",
    "corridor": "FlatCeiling-v0",
    "downstepper": "DownStepper-v0",
    "downstepperv0": "DownStepper-v0",
}

def _normalize_environment_key(value):
    return "".join(ch for ch in value.lower() if ch.isalnum())

def _evogym_env_name(value):
    selected = str(value or "").strip()
    if not selected:
        return "FlatCeiling-v0"
    if selected in EVOGYM_ENVIRONMENT_OPTIONS:
        return selected
    alias = EVOGYM_ENVIRONMENT_ALIASES.get(_normalize_environment_key(selected))
    if alias is not None:
        return alias
    options = ", ".join(EVOGYM_ENVIRONMENT_OPTIONS)
    raise argparse.ArgumentTypeError(
        f"unknown EvoGym environment {selected!r}; use: {options}"
    )

class Config():

    def _get_params(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--out_path",
            required=False,
            default="tmp_out",
            type=str,
            help="path for results files"
        )
        parser.add_argument(
            "--study_name",
            required=False,
            default="defaultstudy",
            type=str,
            help="",
        )
        parser.add_argument(
            "--experiment_name",
            required=False,
            default="defaultexperiment",
            type=str,
            help="Name of the experiment.",
        )
        parser.add_argument(
            "--algorithm",
            required=False,
            default="EA",
            type=str,
            help="",
        )
        parser.add_argument(
            "--population_size",
            required=False,
            default=10,
            type=int,
        )
        parser.add_argument(
            "--offspring_size",
            required=False,
            default=10,
            type=int,
        )
        parser.add_argument(
            "--num_generations",
            required=False,
            default=10,
            type=int,
        )
        parser.add_argument(
            "--tournament_k",
            required=False,
            default=4,
            type=int,
        )
        parser.add_argument(
            "--max_voxels",
            required=False,
            default=36,
            type=int,
            help="",
        )
        parser.add_argument(
            "--cube_face_size",
            required=False,
            default=6,
            type=int,
            help="",
        )
        parser.add_argument(
            "--voxel_types",
            required=False,
            default="withbone",
            type=str,
            help="list of voxel_types config",
        )
        parser.add_argument(
            "--plastic",
            required=False,
            default=0,
            type=int,
            help="0 is not plastic, 1 is plastic",
        )
        parser.add_argument(
            "--env_conditions",
            required=False,
            default='',
            type=str,
            help="params that define environmental conditions and/or task",
        )
        parser.add_argument(
            "--crossover_prob",
            required=False,
            default=1,
            type=float,
        )
        parser.add_argument(
            "--crossover_type",
            required=False,
            default="promoter_aligned_cut_and_splice",
            type=str,
            help=(
                "Crossover operator for EA. "
                "Use 'promoter_aligned_cut_and_splice' for variable-length "
                "promoter-aligned cut-and-splice recombination, "
                "'arithmetic_recombination' for random-alpha weighted genome "
                "averaging, or 'homologous_gene_block_recombination' for similarity-matched "
                "whole-gene recombination. Use 'mutation_only' as a no-crossover "
                "baseline that copies one parent before mutation."
            ),
        )
        parser.add_argument(
            "--mutation_prob",
            required=False,
            default=0.9,
            type=float,
        )
        parser.add_argument(
            "--min_mutation_prob",
            required=False,
            default=0.2,
            type=float,
            help="Lower bound for adaptive mutation probability in EA."
        )
        parser.add_argument(
            "--max_mutation_prob",
            required=False,
            default=1.0,
            type=float,
            help="Upper bound for adaptive mutation probability in EA."
        )
        parser.add_argument(
            "--target_mutation_success_rate",
            required=False,
            default=0.2,
            type=float,
            help="Target offspring improvement rate used to adapt mutation in EA."
        )
        parser.add_argument(
            "--mutation_adaptation_factor",
            required=False,
            default=1.1,
            type=float,
            help="Multiplicative factor for adaptive mutation updates in EA."
        )
        parser.add_argument(
            "--parent_tournament_k",
            required=False,
            default=None,
            type=int,
            help="Tournament size for Pareto parent selection in EA. Defaults to --tournament_k."
        )
        parser.add_argument(
            "--ea_objectives",
            required=False,
            default="fitness,novelty",
            type=str,
            help="Comma-separated objectives for NSGA-style selection in EA."
        )
        parser.add_argument(
            "--novelty_archive_max_size",
            required=False,
            default=100,
            type=int,
            help="Maximum size of the novelty archive used by EA."
        )
        parser.add_argument(
            "--novelty_archive_add_k",
            required=False,
            default=1,
            type=int,
            help="How many top-novel offspring to add to the novelty archive each generation."
        )
        parser.add_argument(
            "--fitness_metric",
            required=False,
            default="displacement",
            type=str,
        )
        parser.add_argument(
            "--generations",
            required=False,
            default="",
            type=str,
            help="list of generations of be analyzed",
        )
        parser.add_argument(
            "--final_gen",
            required=False,
            default="",
            type=str,
            help="last generation to be analyzed"
        )
        parser.add_argument(
            "--experiments",
            required=False,
            default="",
            type=str,
            help="list of experiment_name",
        )
        parser.add_argument(
            "--ustatic",
            required=False,
            default=0.5,
            type=float,
            help="static friction"
        )
        parser.add_argument(
            "--udynamic",
            required=False,
            default=0.2,
            type=float,
            help="dynamic friction"
        )
        parser.add_argument(
            "--evogym_env_name",
            required=False,
            default="",
            type=_evogym_env_name,
            help=(
                "EvoGym world name to load. Empty uses FlatCeiling-v0."
            )
        )
        parser.add_argument(
            "--evogym_displacement_axis",
            required=False,
            default=None,
            type=int,
            choices=[0, 1],
            help="Axis used for the displacement fitness metric. Defaults to y for climbing worlds and x otherwise."
        )
        parser.add_argument(
            "--evogym_left_wall",
            required=False,
            default=0,
            type=int,
            help="If 1, add a fixed terrain wall at the left edge of generated EvoGym worlds."
        )
        parser.add_argument(
            "--evogym_flat_ceiling_gap_blocks",
            required=False,
            default=6,
            type=int,
            help="Empty grid cells between the flat floor and ceiling for FlatCeiling-v0."
        )
        parser.add_argument(
            "--evogym_flat_ceiling_width",
            required=False,
            default=100,
            type=int,
            help="Width in grid cells of generated FlatCeiling-v0 worlds."
        )
        parser.add_argument(
            "--evogym_left_wall_height",
            required=False,
            default=8,
            type=int,
            help="Height in grid cells of the generated EvoGym left wall."
        )
        parser.add_argument(
            "--run",
            required=False,
            default=1,
            type=int,
            help="",
        )
        parser.add_argument(
            "--runs",
            required=False,
            default="",
            type=str,
            help="list of all runs",
        )
        parser.add_argument(
            "--run_simulation",
            required=False,
            default=1,
            type=int,
            help="If 0, runs optimizer without simulating robots, so behavioral measures are none."
        )
        parser.add_argument(
            "--evogym_num_workers",
            required=False,
            default=0,
            type=int,
            help="EvoGym batch workers. 0=auto based on machine CPU."
        )
        parser.add_argument(
            "--evogym_isolate_tasks",
            required=False,
            default=None,
            type=int,
            help=(
                "If 1, evaluate each robot in a fresh worker process. "
                "If 0, reuse workers for better throughput. "
                "Default keeps the platform-specific simulator safety behavior."
            )
        )
        parser.add_argument(
            "--evogym_steps",
            required=False,
            default=500,
            type=int,
            help="Physics steps per robot evaluation in EvoGym."
        )
        parser.add_argument(
            "--evogym_init_x",
            required=False,
            default=3,
            type=int,
            help="Initial robot x position in EvoGym world."
        )
        parser.add_argument(
            "--evogym_init_y",
            required=False,
            default=7,
            type=int,
            help="Initial robot y position in EvoGym world."
        )
        parser.add_argument(
            "--evogym_action_bias",
            required=False,
            default=1.0,
            type=float,
            help="Center value for the EvoGym ANN controller output."
        )
        parser.add_argument(
            "--evogym_action_amplitude",
            required=False,
            default=0.6,
            type=float,
            help="Output amplitude for the EvoGym ANN controller."
        )
        parser.add_argument(
            "--evogym_ann_hidden_size",
            required=False,
            default=8,
            type=int,
            help="Hidden-layer width for the fixed-topology EvoGym ANN controller."
        )
        parser.add_argument(
            "--evogym_sine_period",
            required=False,
            default=40.0,
            type=float,
            help="Oscillator period in simulation steps for the sine-regularized EvoGym ANN controller."
        )
        parser.add_argument(
            "--evogym_sine_mix",
            required=False,
            default=0.35,
            type=float,
            help="Blend factor between ANN output and the built-in sine oscillator. 0=ANN only, 1=sine only."
        )
        parser.add_argument(
            "--evogym_headless",
            required=False,
            default=1,
            type=int,
            help="1=headless (default), 0=render simulation window for debugging."
        )
        parser.add_argument(
            "--evogym_render_mode",
            required=False,
            default="screen",
            type=str,
            help="Render mode when evogym_headless=0 (screen or human)."
        )
        args = parser.parse_args()
        return args
