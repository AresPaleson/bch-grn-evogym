import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT))

try:
    from .voxel_types import VOXEL_TYPES, VOXEL_TYPES_NOBONE, TF_WEIGHTS, TF_WEIGHTS_NOBONE

except ImportError:
    from algorithms.voxel_types import VOXEL_TYPES, VOXEL_TYPES_NOBONE, TF_WEIGHTS, TF_WEIGHTS_NOBONE

class GRN:
    diffusion_sites_qt = 4
    header_nucleotides = 2
    min_genome_length = header_nucleotides

    def __init__(self, promoter_threshold=0.95, max_voxels=25, cube_face_size=5,
                  genotype=None, voxel_types='withbone', env_conditions=None, plastic=None,
                  record_history=False):
        self.max_voxels = max_voxels
        self.genotype = list(genotype or [])
        self.env_conditions = env_conditions
        self.plastic = plastic
        self.cells = []
        self.phenotype = None
        self.genes = []
        self.quantity_voxels = 0
        self.record_history = record_history
        self.development_history = []
        self.regulatory_transcription_factor_idx = 0
        self.regulatory_v1_idx = 1
        self.regulatory_v2_idx = 2
        self.transcription_factor_idx = 3
        self.transcription_factor_amount_idx = 4
        self.diffusion_site_idx = 5
        self.types_nucleotides = 6
        self.promoter_threshold = promoter_threshold
        self.concentration_decay = 0.005
        self.cube_face_size = cube_face_size
        self.structural_products = None
        self.regulatory_products = 2
        if voxel_types == 'withbone':
            self.structural_products = VOXEL_TYPES
            self.tf_weights = TF_WEIGHTS
        elif voxel_types == 'nobone':
            self.structural_products = VOXEL_TYPES_NOBONE
            self.tf_weights = TF_WEIGHTS_NOBONE
        else:
            raise ValueError(f"Unsupported voxel_types: {voxel_types}")
        self.development_products = {
            name: voxel_id
            for name, voxel_id in self.structural_products.items()
            if name != 'offphase_muscle'
        }
        self.structural_tfs = []
        for tf in range(1, len(self.development_products) + 1):
            self.structural_tfs.append(f'TF{tf}')
        self.increase_scaling = 100
        self.intra_diffusion_rate = self.concentration_decay/2
        self.inter_diffusion_rate = self.intra_diffusion_rate/8
        self.dev_steps = 200
        if len(self.genotype) < GRN.header_nucleotides:
            self.genotype.extend([0.0] * (GRN.header_nucleotides - len(self.genotype)))
        self.concentration_threshold = np.minimum(0.1, self.genotype[0])
        self.offphase_alternation_param = float(self.genotype[1])
        self.offphase_alternation_range = [1, 4]
        self._phase_run = 0
        self.genotype = self.genotype[GRN.header_nucleotides:]

    def develop(self):
        k_min, k_max = self.offphase_alternation_range
        span = (k_max - k_min + 1)
        k = k_min + int(self.offphase_alternation_param * span)
        self.offphase_alternation_k = min(k, k_max)
        self.develop_body()
        return self.phenotype

    def develop_body(self):
        self.gene_parser()
        self.regulate()

    def gene_parser(self):
        nucleotide_idx = 0
        while nucleotide_idx < len(self.genotype):
            if self.genotype[nucleotide_idx] < self.promoter_threshold:
                if (len(self.genotype)-1-nucleotide_idx) >= self.types_nucleotides:
                    regulatory_transcription_factor = self.genotype[nucleotide_idx+self.regulatory_transcription_factor_idx+1]
                    regulatory_v1 = self.genotype[nucleotide_idx+self.regulatory_v1_idx+1]
                    regulatory_v2 = self.genotype[nucleotide_idx+self.regulatory_v2_idx+1]
                    transcription_factor = self.genotype[nucleotide_idx+self.transcription_factor_idx+1]
                    transcription_factor_amount = max(0.1, self.genotype[nucleotide_idx+self.transcription_factor_amount_idx+1])
                    diffusion_site = self.genotype[nucleotide_idx+self.diffusion_site_idx+1]
                    limits, _ = self.build_tf_limits(self.development_products,self.regulatory_products, self.tf_weights)
                    regulatory_transcription_factor_label = self.tf_value_to_label(regulatory_transcription_factor, limits)
                    transcription_factor_label = self.tf_value_to_label(transcription_factor, limits)
                    range_size = 1.0 / GRN.diffusion_sites_qt
                    diffusion_site_label = min(int(diffusion_site / range_size), GRN.diffusion_sites_qt - 1)
                    gene = [regulatory_transcription_factor_label, regulatory_v1, regulatory_v2,
                            transcription_factor_label, transcription_factor_amount, diffusion_site_label]
                    self.genes.append(gene)
                    nucleotide_idx += self.types_nucleotides
            nucleotide_idx += 1
        self.genes = np.array(self.genes)

    def build_tf_limits(self, structural_products, regulatory_products, tf_weights):
        weights = []
        for name in structural_products:
            weights.append(float(tf_weights[name]))
        reg_w = float(tf_weights.get("regulatory", 1.0))
        weights.extend([reg_w] * regulatory_products)
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        limits = np.concatenate(([0.0], np.cumsum(weights)))
        limits[-1] = 1.0
        return limits, weights.size

    def tf_value_to_label(self, value, limits):
        v = float(value)
        if v >= 1.0:
            v = np.nextafter(1.0, 0.0)
        elif v < 0.0:
            v = 0.0
        idx = np.searchsorted(limits, v, side="right") - 1
        idx = max(0, min(idx, len(limits) - 2))
        return f"TF{idx + 1}"

    def regulate(self):
        self.phenotype = np.zeros((self.cube_face_size, self.cube_face_size),
                                  dtype=object)
        if len(self.genes) == 0:
            return
        self.maternal_injection()
        self.record_snapshot()
        self.growth()

    def phenotype_to_materials(self):
        materials = np.zeros(self.phenotype.shape, dtype=int)
        for idx, value in np.ndenumerate(self.phenotype):
            materials[idx] = value.voxel_type if value != 0 else 0
        return materials

    def record_snapshot(self):
        if self.record_history:
            self.development_history.append(self.phenotype_to_materials())

    def growth(self):
        maximum_reached = False
        for _ in range(0, self.dev_steps):
            for idxc in range(0, len(self.cells)):
                cell = self.cells[idxc]
                self.increase(cell)
                self.place_voxel(cell)
                if self.quantity_voxels == self.max_voxels -1:
                    maximum_reached = True
                    break
                for tf in cell.transcription_factors:
                    self.decay(tf, cell)
            if maximum_reached:
                break

    def increase(self, cell):
        for idg, gene in enumerate(self.genes):
            if idg in cell.original_genes:
                if cell.transcription_factors.get(gene[self.regulatory_transcription_factor_idx]):
                    tf_in_all_sites = sum(cell.transcription_factors[gene[self.regulatory_transcription_factor_idx]])
                    regulatory_min_val = min(float(gene[self.regulatory_v1_idx]),
                                             float(gene[self.regulatory_v2_idx]))
                    regulatory_max_val = max(float(gene[self.regulatory_v1_idx]),
                                             float(gene[self.regulatory_v2_idx]))
                    if tf_in_all_sites >= regulatory_min_val and tf_in_all_sites <= regulatory_max_val:
                        cell.transcription_factors[gene[self.transcription_factor_idx]][int(gene[self.diffusion_site_idx])] += (
                            float(gene[self.transcription_factor_amount_idx])
                            / float(self.increase_scaling)
                        )

    def decay(self, tf, cell):
        for ds in range(0, GRN.diffusion_sites_qt):
            cell.transcription_factors[tf][ds] = max(
                0,
                cell.transcription_factors[tf][ds] - self.concentration_decay,
            )

    def place_voxel(self, parent_cell):
        product_concentrations = []
        for idm in range(0, len(self.development_products)):
            concentration = (
                sum(parent_cell.transcription_factors[self.structural_tfs[idm]])
                if parent_cell.transcription_factors.get(self.structural_tfs[idm])
                else 0
            )
            product_concentrations.append(concentration)
        idx_max = product_concentrations.index(max(product_concentrations))
        if product_concentrations[idx_max] > self.concentration_threshold:
            freeslots = np.array([c is None for c in parent_cell.children])
            if any(freeslots):
                true_indices = np.where(freeslots)[0]
                values_at_true_indices = np.array(parent_cell.transcription_factors[self.structural_tfs[idx_max]])[true_indices]
                max_value_index = np.argmax(values_at_true_indices)
                position_of_max_value = true_indices[max_value_index]
                slot = position_of_max_value
                potential_child_coord, child_slot = self.find_child_slot(parent_cell.xy_coordinates, slot)
                if all(0 <= i < self.cube_face_size for i in potential_child_coord):
                    if self.phenotype[tuple(potential_child_coord)] == 0:
                        key, voxel_type = list(self.development_products.items())[idx_max]
                        if key == 'phase_muscle':
                            if self._phase_run >= self.offphase_alternation_k:
                                voxel_type = self.structural_products['offphase_muscle']
                                self._phase_run = 0
                            else:
                                self._phase_run += 1
                        self.quantity_voxels += 1
                        self.new_cell(voxel_type, parent_cell, slot, child_slot, potential_child_coord)

    def new_cell(self, voxel_type, parent_cell, parent_slot, child_slot, xy_coordinates, phase_offset=0.0):
        new_cell = Cell(voxel_type=voxel_type, parent_cell=parent_cell,
                        xy_coordinates=xy_coordinates, phase_offset=phase_offset)
        self.phenotype[tuple(xy_coordinates)] = new_cell
        for tf in parent_cell.transcription_factors:
            if parent_cell.transcription_factors[tf][parent_slot] > 0:
                half_concentration = parent_cell.transcription_factors[tf][parent_slot] / 2
                parent_cell.transcription_factors[tf][parent_slot] = half_concentration
                new_cell.transcription_factors[tf] = [0] * GRN.diffusion_sites_qt
                new_cell.transcription_factors[tf][child_slot] = half_concentration
        self.express_genes(new_cell)
        self.cells.append(new_cell)
        self.record_snapshot()

    def find_child_slot(self, xy_coordinates_parent, parent_slot):
        x = 0
        y = 1
        if parent_slot == DS.LEFT:
            child_slot = DS.RIGHT
            xy_coordinates_child = list(xy_coordinates_parent)
            xy_coordinates_child[x] -= 1
        if parent_slot == DS.RIGHT:
            child_slot = DS.LEFT
            xy_coordinates_child = list(xy_coordinates_parent)
            xy_coordinates_child[x] += 1
        if parent_slot == DS.UP:
            child_slot = DS.DOWN
            xy_coordinates_child = list(xy_coordinates_parent)
            xy_coordinates_child[y] += 1
        if parent_slot == DS.DOWN:
            child_slot = DS.UP
            xy_coordinates_child = list(xy_coordinates_parent)
            xy_coordinates_child[y] -= 1
        return xy_coordinates_child, child_slot

    def maternal_injection(self):
        first_gene_idx = 0
        tf_label_idx = 0
        min_value_idx = 1
        mother_tf_label = self.genes[first_gene_idx][tf_label_idx]
        mother_tf_injection = float(self.genes[first_gene_idx][min_value_idx])
        middle_pos = [s // 2 for s in self.phenotype.shape]
        first_cell = Cell(voxel_type=self.structural_products['offphase_muscle'],
                          parent_cell=None,
                          xy_coordinates=middle_pos)
        first_cell.xy_coordinates = middle_pos
        first_cell.transcription_factors[mother_tf_label] = (
            [mother_tf_injection/GRN.diffusion_sites_qt] * GRN.diffusion_sites_qt
        )
        self.express_genes(first_cell)
        self.cells.append(first_cell)
        self.phenotype[tuple(middle_pos)] = first_cell

    def express_genes(self, new_cell):
        for idg, gene in enumerate(self.genes):
            regulatory_min_val = min(float(gene[self.regulatory_v1_idx]),
                                     float(gene[self.regulatory_v2_idx]))
            regulatory_max_val = max(float(gene[self.regulatory_v1_idx]),
                                     float(gene[self.regulatory_v2_idx]))
            if new_cell.transcription_factors.get(gene[self.regulatory_transcription_factor_idx]):
                tf_in_all_sites = sum(new_cell.transcription_factors[gene[self.regulatory_transcription_factor_idx]])
                if tf_in_all_sites >= regulatory_min_val and tf_in_all_sites <= regulatory_max_val:
                    if new_cell.transcription_factors.get(gene[self.transcription_factor_idx]):
                        new_cell.transcription_factors[gene[self.transcription_factor_idx]][
                            int(gene[self.diffusion_site_idx])
                        ] += float(gene[self.transcription_factor_amount_idx])
                    else:
                        new_cell.transcription_factors[gene[self.transcription_factor_idx]] = [0] * GRN.diffusion_sites_qt
                        new_cell.transcription_factors[gene[self.transcription_factor_idx]][
                            int(gene[self.diffusion_site_idx])
                        ] = float(gene[self.transcription_factor_amount_idx])
                    new_cell.original_genes.append(idg)

class Cell:

    def __init__(self, voxel_type, parent_cell, xy_coordinates, phase_offset=0.0):
        self.voxel_type = voxel_type
        self.phase_offset = phase_offset
        self.transcription_factors = {}
        self.original_genes = []
        self.xy_coordinates = xy_coordinates
        self.parent_cell = parent_cell
        self.children = [None] * GRN.diffusion_sites_qt

class DS:
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

def initialization(rng, ini_genome_size):
    genome_ini_size = ini_genome_size
    genome_size = genome_ini_size + GRN.header_nucleotides
    genotype = [round(rng.uniform(0, 1), 2) for _ in range(genome_size)]
    return genotype

def promoter_aligned_cut_and_splice_crossover(
        rng,
        promoter_threshold,
        max_geno_size,
        parent1,
        parent2,
):
    parent1 = parent1.genome
    parent2 = parent2.genome
    types_nucleotides = 6
    header_size = GRN.header_nucleotides
    if len(parent1) < header_size or len(parent2) < header_size:
        fallback = list(parent1 if len(parent1) >= len(parent2) else parent2)
        return fallback[:max_geno_size]
    new_genotype = [
        (parent1[0] + parent2[0]) / 2,
        (parent1[1] + parent2[1]) / 2,
    ]
    p1 = parent1[header_size:]
    p2 = parent2[header_size:]

    def get_promoters(parent):
        promotor_sites = []
        nucleotide_idx = 0
        while nucleotide_idx < len(parent):
            if parent[nucleotide_idx] < promoter_threshold:
                if (len(parent) - 1 - nucleotide_idx) >= types_nucleotides:
                    promotor_sites.append(nucleotide_idx)
                    nucleotide_idx += types_nucleotides
            nucleotide_idx += 1
        return promotor_sites
    promoters_p1 = get_promoters(p1)
    if promoters_p1:
        cut_p1 = rng.sample(promoters_p1, 1)[0]
        take_head_p1 = rng.random() < 0.5
        if take_head_p1:
            subset_p1 = p1[0:cut_p1 + types_nucleotides + 1]
        else:
            subset_p1 = p1[cut_p1:]
    else:
        subset_p1 = []
    new_genotype += subset_p1
    prop_from_p1 = (len(subset_p1) / len(p1)) if len(p1) > 0 else 0.0
    promoters_p2 = get_promoters(p2)
    target_prop_p2 = 1.0 - prop_from_p1
    target_len_p2 = int(round(target_prop_p2 * len(p2))) if len(p2) > 0 else 0
    take_head_p2 = rng.random() < 0.5
    if promoters_p2 and len(p2) > 0:
        best_cut = None
        best_diff = None
        for c in promoters_p2:
            if take_head_p2:
                seg_len = min(c + types_nucleotides + 1, len(p2))
            else:
                seg_len = len(p2) - c
            diff = abs(seg_len - target_len_p2)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_cut = c
        cut_p2 = best_cut if best_cut is not None else promoters_p2[0]
        if take_head_p2:
            subset_p2 = p2[0: min(cut_p2 + types_nucleotides + 1, len(p2))]
        else:
            subset_p2 = p2[cut_p2:]
    else:
        subset_p2 = []
    new_genotype += subset_p2
    if len(new_genotype) > max_geno_size:
        new_genotype = new_genotype[:max_geno_size]
    if len(new_genotype) < header_size:
        new_genotype.extend([0.0] * (header_size - len(new_genotype)))
    return new_genotype

def _gene_blocks(parent, promoter_threshold, types_nucleotides):
    gene_blocks = []
    nucleotide_idx = 0
    while nucleotide_idx < len(parent):
        if parent[nucleotide_idx] < promoter_threshold:
            if (len(parent) - 1 - nucleotide_idx) >= types_nucleotides:
                gene_blocks.append(parent[nucleotide_idx:nucleotide_idx + types_nucleotides + 1])
                nucleotide_idx += types_nucleotides
        nucleotide_idx += 1
    return gene_blocks

def _gene_block_records(parent, promoter_threshold, types_nucleotides):
    gene_blocks = []
    nucleotide_idx = 0
    while nucleotide_idx < len(parent):
        if parent[nucleotide_idx] < promoter_threshold:
            if (len(parent) - 1 - nucleotide_idx) >= types_nucleotides:
                gene_blocks.append(
                    {
                        "start": nucleotide_idx,
                        "block": parent[nucleotide_idx:nucleotide_idx + types_nucleotides + 1],
                    }
                )
                nucleotide_idx += types_nucleotides
        nucleotide_idx += 1
    return gene_blocks

def _block_distance(block1, block2):
    if not block1 or not block2:
        return float("inf")
    length = min(len(block1), len(block2))
    return sum(abs(float(block1[idx]) - float(block2[idx])) for idx in range(length)) / length

def _finite_fitness(individual):
    value = getattr(individual, "fitness", None)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value

def homologous_gene_block_recombination_crossover(
        rng,
        promoter_threshold,
        max_geno_size,
        parent1,
        parent2,
        match_threshold=0.35,
        fitter_parent_bias=0.70,
        secondary_innovation_prob=0.15,
):
    genome1 = parent1.genome
    genome2 = parent2.genome
    types_nucleotides = 6
    header_size = GRN.header_nucleotides
    if len(genome1) < header_size or len(genome2) < header_size:
        fallback = list(genome1 if len(genome1) >= len(genome2) else genome2)
        return fallback[:max_geno_size]
    fitness1 = _finite_fitness(parent1)
    fitness2 = _finite_fitness(parent2)
    if fitness1 is not None and fitness2 is not None and fitness1 != fitness2:
        primary_genome, secondary_genome = (
            (genome1, genome2) if fitness1 > fitness2 else (genome2, genome1)
        )
    else:
        primary_genome, secondary_genome = (
            (genome1, genome2) if rng.random() < 0.5 else (genome2, genome1)
        )
    primary_blocks = _gene_block_records(
        primary_genome[header_size:], promoter_threshold, types_nucleotides
    )
    secondary_blocks = _gene_block_records(
        secondary_genome[header_size:], promoter_threshold, types_nucleotides
    )
    if not primary_blocks:
        fallback = list(secondary_genome if secondary_blocks else primary_genome)
        return fallback[:max_geno_size]
    new_genotype = [
        (float(genome1[0]) + float(genome2[0])) / 2,
        (float(genome1[1]) + float(genome2[1])) / 2,
    ]
    candidate_pairs = []
    for primary_idx, primary_record in enumerate(primary_blocks):
        for secondary_idx, secondary_record in enumerate(secondary_blocks):
            distance = _block_distance(primary_record["block"], secondary_record["block"])
            if distance <= match_threshold:
                candidate_pairs.append((distance, primary_idx, secondary_idx))
    candidate_pairs.sort(key=lambda item: item[0])
    matched_primary = {}
    used_primary = set()
    used_secondary = set()
    for _, primary_idx, secondary_idx in candidate_pairs:
        if primary_idx in used_primary or secondary_idx in used_secondary:
            continue
        matched_primary[primary_idx] = secondary_idx
        used_primary.add(primary_idx)
        used_secondary.add(secondary_idx)
    secondary_insertions = [
        record["block"]
        for idx, record in enumerate(secondary_blocks)
        if idx not in used_secondary and rng.random() < secondary_innovation_prob
    ]
    for primary_idx, primary_record in enumerate(primary_blocks):
        secondary_idx = matched_primary.get(primary_idx)
        if secondary_idx is None:
            selected_block = primary_record["block"]
        else:
            take_primary = rng.random() < fitter_parent_bias
            selected_block = (
                primary_record["block"]
                if take_primary
                else secondary_blocks[secondary_idx]["block"]
            )
        new_genotype += selected_block
        if secondary_insertions and rng.random() < secondary_innovation_prob:
            new_genotype += secondary_insertions.pop(0)
        if len(new_genotype) >= max_geno_size:
            return new_genotype[:max_geno_size]
    for block in secondary_insertions:
        if len(new_genotype) + len(block) > max_geno_size:
            break
        new_genotype += block
    if len(new_genotype) > max_geno_size:
        new_genotype = new_genotype[:max_geno_size]
    if len(new_genotype) < header_size:
        new_genotype.extend([0.0] * (header_size - len(new_genotype)))
    return new_genotype

def arithmetic_recombination_crossover(
        rng,
        max_geno_size,
        parent1,
        parent2,
):
    parent1 = parent1.genome
    parent2 = parent2.genome
    child_size = min(len(parent1), len(parent2), max_geno_size)
    if child_size <= 0:
        return []
    alpha = rng.uniform(0, 1)
    return [
        alpha * float(parent1[idx]) + (1.0 - alpha) * float(parent2[idx])
        for idx in range(child_size)
    ]

def mutation_type1(rng, genome):
    genome = list(genome)
    header_size = GRN.header_nucleotides
    min_length = GRN.min_genome_length
    if len(genome) < min_length:
        genome.extend([round(rng.uniform(0, 1), 2) for _ in range(min_length - len(genome))])
    position = rng.sample(range(0, len(genome)), 1)[0]
    type = rng.sample(['perturbation', 'deletion', 'addition', 'swap'], 1)[0]
    if type == 'perturbation':
        newv = round(genome[position] + rng.normalvariate(0, 0.1), 2)
        if newv > 1:
            genome[position] = 1
        elif newv < 0:
            genome[position] = 0
        else:
            genome[position] = newv
    if type == 'deletion' and position >= header_size and len(genome) > min_length:
        genome.pop(position)
    if type == 'addition':
        insert_at = max(header_size, position)
        genome.insert(insert_at, round(rng.uniform(0, 1), 2))
    if type == 'swap':
        position2 = rng.sample(range(0, len(genome)), 1)[0]
        while position == position2:
            position2 = rng.sample(range(0, len(genome)), 1)[0]
        if (position < header_size) != (position2 < header_size):
            return genome
        position_v = genome[position]
        position2_v = genome[position2]
        genome[position] = position2_v
        genome[position2] = position_v
    return genome
