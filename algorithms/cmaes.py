import os
import sys
import numpy as np
from pathlib import Path
import shutil
import cma
import time

ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT))

from algorithms.experiment import Experiment
from algorithms.EA_classes import Individual
from algorithms.GRN import GRN, initialization
from simulation.simulation_resources import simulate_evogym_batch
from simulation.prepare_robot_files import prepare_robot_files
from utils.metrics import genopheno_abs_metrics, relative_metrics
from utils.config import Config


class CMAES(Experiment):

    def __init__(self, args=None):
        self.args = args if args is not None else Config()._get_params()
        super().__init__(self.args)
        self.PROMOTOR_THRESHOLD = 0.95
        self.GENOME_SIZE = 300
        self.N = self.GENOME_SIZE
        self.sigma0 = 0.2
        self.cube_face_size = self.args.cube_face_size
        self.max_voxels = self.args.max_voxels
        self.voxel_types = self.args.voxel_types
        self.plastic = self.args.plastic
        self.env_conditions = self.args.env_conditions
        self.population_size = self.args.population_size
        self.num_generations = self.args.num_generations
        self.fitness_metric = self.args.fitness_metric
        self.cma_opts = None
        self.es = None

    def develop_phenotype(self, genome, voxel_types):
        phenotype = GRN(
            promoter_threshold=self.PROMOTOR_THRESHOLD,
            max_voxels=self.max_voxels,
            cube_face_size=self.cube_face_size,
            voxel_types=voxel_types,
            genotype=genome,
            env_conditions=self.env_conditions,
            plastic=self.plastic,
        ).develop()
        phenotype_materials = np.zeros(phenotype.shape, dtype=int)
        for index, value in np.ndenumerate(phenotype):
            phenotype_materials[index] = value.voxel_type if value != 0 else 0
        return phenotype_materials

    def vector_to_individual(self, x, generation):
        x = np.asarray(x, dtype=float)
        genome = list(x)
        self.id_counter += 1
        ind = Individual(genome, self.id_counter, parent1_id=None, parent2_id=None)
        ind.born_generation = generation
        return ind

    def run(self):
        super().recover_db()
        last_gen, recovered_population = self._recover_state()
        self.cma_opts = {
            "popsize": self.population_size,
            "seed": int(self.rng.randint(1, 2**31 - 1)),
            "bounds": [0.0, 1.0],
        }
        if recovered_population is None:
            start_gen = 1
            x0 = np.asarray(initialization(self.rng, self.GENOME_SIZE), dtype=float)
            self.es = cma.CMAEvolutionStrategy(x0, self.sigma0, self.cma_opts)
        else:
            start_gen = last_gen + 1
            best = max(recovered_population, key=lambda ind: ind.fitness)
            g = np.asarray(best.genome, dtype=float)
            if g.size != self.N:
                x0 = np.asarray(initialization(self.rng, self.GENOME_SIZE), dtype=float)
            else:
                eps = 1e-9
                g = np.clip(g, eps, 1.0 - eps)
                x0 = np.log(g / (1.0 - g))
            self.es = cma.CMAEvolutionStrategy(x0, self.sigma0, self.cma_opts)
            print(
                f"Recovered last completed generation = {last_gen}, "
                f"population size = {len(recovered_population)}, next id = {self.id_counter + 1}"
            )
        if start_gen > self.num_generations:
            print(f"Nothing to do: start_gen={start_gen} > num_generations={self.num_generations}")
            return
        for generation in range(start_gen, self.num_generations + 1):
            xs = self.es.ask()
            individuals = [self.vector_to_individual(x, generation) for x in xs]
            for ind in individuals:
                ind.phenotype = self.develop_phenotype(ind.genome, self.voxel_types)
                genopheno_abs_metrics(ind, self.args)
                if self.args.run_simulation:
                    prepare_robot_files(ind, self.args)
            if self.args.run_simulation:
                simulate_evogym_batch(individuals, self.args)
            relative_metrics(individuals, self.args, generation)
            losses = [-float(ind.fitness) for ind in individuals]
            self.es.tell(xs, losses)
            individuals_sorted = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
            survivors = individuals_sorted[: self.population_size]
            self._persist_generation_atomic(generation, individuals, survivors)
            print(
                f"Finished generation {generation}. Best fitness: {survivors[0].fitness:.4f}",
                flush=True,
            )
        try:
            self.session.close()
        except Exception:
            pass
        path_robots = f"{self.args.out_path}/{self.args.study_name}/{self.args.experiment_name}/run_{self.args.run}/robots"
        if os.path.exists(path_robots):
            shutil.rmtree(path_robots)
        print("Finished optimizing.")

if __name__ == "__main__":
    start = time.time()
    CMAES().run()
    end = time.time()
    elapsed = end - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"\n[RUN-TIME]  {hours}h {minutes}m {seconds:.1f}s")
