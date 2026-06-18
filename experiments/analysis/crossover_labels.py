"""Shared display labels for crossover operator plots."""

CROSSOVER_ORDER = (
    "promoter_aligned_cut_and_splice",
    "arithmetic_recombination",
    "homologous_gene_block_recombination",
)

CROSSOVER_LABELS = {
    "promoter_aligned_cut_and_splice": "cut and splice",
    "arithmetic_recombination": "arithmetic recombination",
    "homologous_gene_block_recombination": "homologous recombination",
}

CROSSOVER_COLORS = {
    "promoter_aligned_cut_and_splice": "#4E79A7",
    "arithmetic_recombination": "#F28E2B",
    "homologous_gene_block_recombination": "#59A14F",
}

CROSSOVER_ALIASES = {
    "cutpoint": "promoter_aligned_cut_and_splice",
    "cut_and_splice": "promoter_aligned_cut_and_splice",
    "one_point": "promoter_aligned_cut_and_splice",
    "old": "promoter_aligned_cut_and_splice",
    "proportional": "promoter_aligned_cut_and_splice",
    "unequal_prop": "promoter_aligned_cut_and_splice",
    "blended": "arithmetic_recombination",
    "blend": "arithmetic_recombination",
    "weighted_average": "arithmetic_recombination",
    "weighted": "arithmetic_recombination",
    "arithmetic": "arithmetic_recombination",
    "arithmetic_crossover": "arithmetic_recombination",
    "intermediate_recombination": "arithmetic_recombination",
    "homologous_gene_block": "homologous_gene_block_recombination",
    "homologous_gene_recombination": "homologous_gene_block_recombination",
    "homologous": "homologous_gene_block_recombination",
    "aligned_gene_block": "homologous_gene_block_recombination",
    "gene_block": "homologous_gene_block_recombination",
}


def strip_standard_experiment_prefix(experiment: str) -> str:
    label = str(experiment)
    if label.startswith("flat_"):
        label = label[len("flat_"):]
    return label


def infer_crossover_type(experiment: str) -> str:
    label = strip_standard_experiment_prefix(experiment)
    candidates = sorted(
        list(CROSSOVER_ORDER) + list(CROSSOVER_ALIASES),
        key=len,
        reverse=True,
    )
    for candidate in candidates:
        if label == candidate or label.startswith(f"{candidate}_"):
            return CROSSOVER_ALIASES.get(candidate, candidate)
    return str(experiment)


def display_crossover_name(crossover_type: str) -> str:
    canonical = CROSSOVER_ALIASES.get(str(crossover_type), str(crossover_type))
    return CROSSOVER_LABELS.get(canonical, str(crossover_type).replace("_", " "))


def display_experiment_name(experiment: str) -> str:
    return display_crossover_name(infer_crossover_type(experiment))
