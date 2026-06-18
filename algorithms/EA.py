import os
import sys
import math
from pathlib import Path
import shutil
import time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT))

from algorithms.experiment import Experiment
from algorithms.EA_classes import Individual
from algorithms.GRN import (
    GRN,
    arithmetic_recombination_crossover,
    homologous_gene_block_recombination_crossover,
    initialization,
    mutation_type1,
    promoter_aligned_cut_and_splice_crossover,
)

from simulation.simulation_resources import simulate_evogym_batch
from simulation.prepare_robot_files import prepare_robot_files
from utils.metrics import genopheno_abs_metrics
from utils.config import Config

class EA(Experiment):
    """
    Small elitist genetic algorithm.
    Selection is intentionally direct: fitness wins. Novelty/NSGA bookkeeping is
    left out of the search loop so locomotion pressure stays strong.
    """

    def __init__(self, args=None):
        self.args = args if args is not None else Config()._get_params()
        super().__init__(self.args)
        self.MAX_GENOME_SIZE = 1000
        self.INI_GENOME_SIZE = 300
        self.PROMOTOR_THRESHOLD = 0.95
        self.cube_face_size = self.args.cube_face_size
        self.max_voxels = self.args.max_voxels
        self.voxel_types = self.args.voxel_types
        self.plastic = self.args.plastic
        self.env_conditions = self.args.env_conditions
        self.population_size = int(self.args.population_size)
        self.offspring_size = int(self.args.offspring_size)
        self.crossover_prob = float(self.args.crossover_prob)
        self.crossover_type = self._normalize_crossover_type(
            getattr(self.args, "crossover_type", "promoter_aligned_cut_and_splice")
        )
        self.mutation_prob = float(self.args.mutation_prob)
        self.tournament_k = max(2, int(self.args.tournament_k))
        self.num_generations = int(self.args.num_generations)
        self.fitness_metric = self.args.fitness_metric
        self.elitism = max(1, int(getattr(self.args, "elitism", 3)))
        self.evaluation_cache = {}

    def _normalize_crossover_type(self, crossover_type):
        aliases = {
            "promoter_aligned_cut_and_splice": "promoter_aligned_cut_and_splice",
            "cut_and_splice": "promoter_aligned_cut_and_splice",
            "cutpoint": "promoter_aligned_cut_and_splice",
            "one_point": "promoter_aligned_cut_and_splice",
            "old": "promoter_aligned_cut_and_splice",
            "proportional": "promoter_aligned_cut_and_splice",
            "unequal_prop": "promoter_aligned_cut_and_splice",
            "arithmetic_recombination": "arithmetic_recombination",
            "arithmetic_crossover": "arithmetic_recombination",
            "intermediate_recombination": "arithmetic_recombination",
            "blended": "arithmetic_recombination",
            "blend": "arithmetic_recombination",
            "weighted_average": "arithmetic_recombination",
            "weighted": "arithmetic_recombination",
            "arithmetic": "arithmetic_recombination",
            "homologous_gene_block_recombination": "homologous_gene_block_recombination",
            "homologous_gene_recombination": "homologous_gene_block_recombination",
            "homologous_gene_block": "homologous_gene_block_recombination",
            "homologous": "homologous_gene_block_recombination",
            "aligned_gene_block": "homologous_gene_block_recombination",
            "gene_block": "homologous_gene_block_recombination",
            "mutation_only": "mutation_only",
            "mutation": "mutation_only",
            "no_crossover": "mutation_only",
            "asexual": "mutation_only",
        }
        key = str(crossover_type).strip().lower()
        if key not in aliases:
            valid = ", ".join(sorted(aliases))
            raise ValueError(f"Unsupported crossover_type '{crossover_type}'. Use one of: {valid}")
        return aliases[key]

    def develop_phenotype(self, genome, voxel_types):
        cells = GRN(
            promoter_threshold=self.PROMOTOR_THRESHOLD,
            max_voxels=self.max_voxels,
            cube_face_size=self.cube_face_size,
            voxel_types=voxel_types,
            genotype=genome,
            env_conditions=self.env_conditions,
            plastic=self.plastic,
        ).develop()
        phenotype = np.zeros(cells.shape, dtype=int)
        for index, value in np.ndenumerate(cells):
            phenotype[index] = value.voxel_type if value != 0 else 0
        return phenotype

    def initialize_population(self, size, generation):
        individuals = []
        for _ in range(size):
            self.id_counter += 1
            ind = Individual(
                initialization(self.rng, self.INI_GENOME_SIZE),
                self.id_counter,
                parent1_id=None,
                parent2_id=None,
            )
            ind.born_generation = generation
            individuals.append(ind)
        return individuals

    def _fitness_value(self, individual):
        value = getattr(individual, "fitness", None)
        if value is None:
            return float("-inf")
        try:
            value = float(value)
        except (ValueError, TypeError):
            return float("-inf")
        if not math.isfinite(value):
            return float("-inf")
        age = getattr(individual, 'age', 1)
        if age > 10:
            penalty_factor = min(age, 100) * 0.001
            value -= abs(value) * penalty_factor
        return value

    def _sort_by_fitness(self, population):
        return sorted(population, key=self._fitness_value, reverse=True)

    def _set_relative_metrics(self, population, generation):
        for ind in population:
            ind.age = generation - ind.born_generation + 1
            ind.uniqueness = 0.0
            ind.novelty = 0.0
            displacement = self._metric_value(ind, "displacement")
            ind.novelty_weighted = displacement if math.isfinite(displacement) else float("-inf")
            ind.dominated_disp_nov = 0.0
            ind.fitness = self._fitness_value_from_metric(ind)

    def _fitness_value_from_metric(self, individual):
        return self._metric_value(individual, self.fitness_metric)

    def _metric_value(self, individual, metric):
        value = getattr(individual, metric, None)
        if value is None:
            return float("-inf")
        try:
            value = float(value)
        except (ValueError, TypeError):
            return float("-inf")
        return value if math.isfinite(value) else float("-inf")

    def evaluate_individuals(self, individuals):
        to_simulate = []
        for ind in individuals:
            ind.phenotype = self.develop_phenotype(ind.genome, self.voxel_types)
            genopheno_abs_metrics(ind, self.args)
            if getattr(ind, "valid", True):
                cache_key = tuple(float(gene) for gene in ind.genome)
                if cache_key in self.evaluation_cache:
                    ind.displacement = self.evaluation_cache[cache_key]
                elif self.args.run_simulation:
                    prepare_robot_files(ind, self.args)
                    to_simulate.append(ind)
        if self.args.run_simulation and to_simulate:
            simulate_evogym_batch(to_simulate, self.args)
            for ind in to_simulate:
                displacement = self._metric_value(ind, "displacement")
                if math.isfinite(displacement):
                    cache_key = tuple(float(gene) for gene in ind.genome)
                    self.evaluation_cache[cache_key] = displacement

    def mutate(self, individual):
        if self.rng.uniform(0, 1) <= self.mutation_prob:
            individual.genome = mutation_type1(self.rng, individual.genome)

    def crossover(self, parent1, parent2):
        source_parent = None
        if self.crossover_type == "mutation_only":
            source_parent = self.rng.choice((parent1, parent2))
            child_genome = list(source_parent.genome)
        elif self.rng.uniform(0, 1) <= self.crossover_prob:
            if self.crossover_type == "arithmetic_recombination":
                child_genome = arithmetic_recombination_crossover(
                    self.rng,
                    self.MAX_GENOME_SIZE,
                    parent1,
                    parent2,
                )
            elif self.crossover_type == "homologous_gene_block_recombination":
                child_genome = homologous_gene_block_recombination_crossover(
                    self.rng,
                    self.PROMOTOR_THRESHOLD,
                    self.MAX_GENOME_SIZE,
                    parent1,
                    parent2,
                )
            else:
                child_genome = promoter_aligned_cut_and_splice_crossover(
                    self.rng,
                    self.PROMOTOR_THRESHOLD,
                    self.MAX_GENOME_SIZE,
                    parent1,
                    parent2,
                )
        else:
            source_parent = self.rng.choice((parent1, parent2))
            child_genome = list(source_parent.genome)
        self.id_counter += 1
        if source_parent is not None:
            return Individual(
                child_genome,
                self.id_counter,
                parent1_id=source_parent.id,
                parent2_id=None,
            )
        return Individual(child_genome, self.id_counter, parent1_id=parent1.id, parent2_id=parent2.id)

    def tournament_selection(self, population, exclude_id=None):
        k = min(self.tournament_k, len(population))
        contestants = self.rng.sample(population, k)
        valid_contestants = [
            c for c in contestants
            if c.id != exclude_id and math.isfinite(self._fitness_value(c))
        ]
        if not valid_contestants:
            fallback_pool = [ind for ind in population if ind.id != exclude_id and math.isfinite(self._fitness_value(ind))]
            if not fallback_pool:
                 fallback_pool = population
            return max(fallback_pool, key=self._fitness_value)
        return max(valid_contestants, key=self._fitness_value)

    def make_offspring(self, population, generation):
        offspring = []
        for _ in range(self.offspring_size):
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population, exclude_id=parent1.id)
            child = self.crossover(parent1, parent2)
            child.born_generation = generation
            self.mutate(child)
            offspring.append(child)
        return offspring

    def select_survivors(self, pool):
        return self._sort_by_fitness(pool)[: self.population_size]

    def _print_generation_summary(self, generation, population):
        ranked = self._sort_by_fitness(population)
        best = ranked[0]
        finite = [self._fitness_value(ind) for ind in ranked if math.isfinite(self._fitness_value(ind))]
        mean = float(np.mean(finite)) if finite else float("-inf")
        valid = int(sum(1 for ind in population if getattr(ind, "valid", 0)))
        print(
            f"Finished generation {generation}. "
            f"Best {self.fitness_metric}: {self._fitness_value(best):.4f}. "
            f"Mean finite fitness: {mean:.4f}. "
            f"Valid: {valid}/{len(population)}."
        )

    def _stop_file_path(self):
        return Path(self.out_path) / "STOP"

    def _manual_stop_requested(self):
        stop_path = self._stop_file_path()
        if stop_path.exists():
            print(f"Manual stop requested by stop file: {stop_path}")
            return True
        return False

    def run(self):
        super().recover_db()
        last_gen, recovered_population = self._recover_state()
        stopped_manually = False
        try:
            if recovered_population is None:
                generation = 1
                population = self.initialize_population(self.population_size, generation)
                self.evaluate_individuals(population)
                self._set_relative_metrics(population, generation)
                population = self.select_survivors(population)
                self._persist_generation_atomic(generation, population, population)
                self._print_generation_summary(generation, population)
                start_gen = generation + 1
            else:
                population = recovered_population
                start_gen = last_gen + 1
                print(
                    f"Recovered last completed generation = {last_gen}, "
                    f"population size = {len(population)}, next id = {self.id_counter + 1}"
                )
            if self._manual_stop_requested():
                stopped_manually = True
            for generation in range(start_gen, self.num_generations + 1):
                if self._manual_stop_requested():
                    stopped_manually = True
                    break
                offspring = self.make_offspring(population, generation)
                self.evaluate_individuals(offspring)
                pool = population + offspring
                self._set_relative_metrics(pool, generation)
                population = self.select_survivors(pool)
                self._set_relative_metrics(population, generation)
                self._persist_generation_atomic(generation, offspring, population)
                self._print_generation_summary(generation, population)
                if self._manual_stop_requested():
                    stopped_manually = True
                    break
        except KeyboardInterrupt:
            stopped_manually = True
            print(
                "\nManual stop received. The last fully saved generation can be "
                "resumed by running the script again."
            )
        finally:
            try:
                self.session.close()
            except Exception:
                pass
            path_robots = f"{self.args.out_path}/{self.args.study_name}/{self.args.experiment_name}/run_{self.args.run}/robots"
            if os.path.exists(path_robots):
                shutil.rmtree(path_robots)
        if stopped_manually:
            print("Stopped optimizing manually.")
        else:
            print("Finished optimizing.")

if __name__ == "__main__":
    start = time.time()
    EA().run()
    end = time.time()
    elapsed = end - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"\n[RUN-TIME]  {hours}h {minutes}m {seconds:.1f}s")
