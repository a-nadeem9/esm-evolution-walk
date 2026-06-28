"""
ESM2 evolution walk

Starts from random sequences, proposes mutations, 
Scores them with ESM2-based PRCS plus epistasis, 
Adaptively makes mutation steps smaller as PRCS improves

"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
import pickle
import random
import sys
import time
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", message=".*libomp.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

try:
    import esm
    import torch
    import torch.nn.functional as F
    from scipy.spatial.distance import jensenshannon

    ML_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    esm = None
    torch = None
    F = None
    jensenshannon = None
    ML_IMPORT_ERROR = exc


def no_grad():
    """Use torch.no_grad when torch exists; otherwise let --help still work."""

    if torch is None:
        return lambda func: func
    return torch.no_grad()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global constants
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

ESM2_MODEL = "esm2_t33_650M_UR50D"
REPR_LAYER = 33

# Simulated annealing: start permissive, then cool down.
INITIAL_TEMP = 0.3

# Epistasis weight limits.
LAMBDA_MIN = 0.1
LAMBDA_CAP = 0.5
LAMBDA_COUPLING_SCALE = 5.0
LAMBDA_RAMP_FRAC = 0.4

# Coupling-map settings.
COUPLING_THRESHOLD = 0.05
COUPLING_MIN_FRAC = 0.20

# Main walk settings.
RUNS_PER_PROTEIN = 1
MIN_STEPS = 2000
STEPS_PER_RESIDUE = 10
PROPOSALS_PER_STEP = 40
COUPLING_RECOMPUTE_EVERY = 150
CHECKPOINT_EVERY = 200
LOG_EVERY = 50
PLL_DIAGNOSTIC_EVERY = 500

# Adaptive refinement settings.
# ADAPTIVE_WINDOW checks whether PRCS improved over the recent 400 step window.
ADAPTIVE_WINDOW = 400
ADAPTIVE_PRCS_EPS = 0.005
ADAPTIVE_STAGE1 = 0.50
ADAPTIVE_STAGE2 = 0.60
ADAPTIVE_STAGE3 = 0.75
ADAPTIVE_PLATEAU = 0.35

SAVE_SEQUENCE_TRAJECTORY = True
OVERWRITE_COMPLETED = False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Configuration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass(frozen=True)
class WalkConfig:
    k_single: int
    group_size_min: int
    group_size_max: int
    n_groups: int
    p_coupled: float
    max_steps: int
    cooling_rate: float


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Command-line inputs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# User only gives input folder, output folder, optional protein subset & seed.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an adaptive ESM2 evolution walk from random sequence to WT FASTA targets."
    )

    parser.add_argument(
        "--fasta-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "target_fastas",
        help="Folder containing .fasta or .fa wild-type target sequences.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("evolution_walk_output"),
        help="Output folder for trajectories, summaries, FASTAs, and checkpoints.",
    )

    parser.add_argument(
        "--protein-filter",
        default="",
        help="Optional comma-separated list of FASTA stems to run, e.g. gapdh,ighg1.",
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Fixed method settings. These are constants, not user-facing CLI params.
    args.runs_per_protein = RUNS_PER_PROTEIN
    args.device = "auto"
    args.model = ESM2_MODEL
    args.min_steps = MIN_STEPS
    args.steps_per_residue = STEPS_PER_RESIDUE
    args.proposals_per_step = PROPOSALS_PER_STEP
    args.coupling_recompute_every = COUPLING_RECOMPUTE_EVERY
    args.checkpoint_every = CHECKPOINT_EVERY
    args.log_every = LOG_EVERY
    args.pll_diagnostic_every = PLL_DIAGNOSTIC_EVERY
    args.adaptive_window = ADAPTIVE_WINDOW
    args.adaptive_prcs_eps = ADAPTIVE_PRCS_EPS
    args.adaptive_stage1 = ADAPTIVE_STAGE1
    args.adaptive_stage2 = ADAPTIVE_STAGE2
    args.adaptive_stage3 = ADAPTIVE_STAGE3
    args.adaptive_plateau = ADAPTIVE_PLATEAU
    args.no_sequence_trajectory = not SAVE_SEQUENCE_TRAJECTORY
    args.overwrite = OVERWRITE_COMPLETED
    return args


def ensure_ml_dependencies() -> None:
    """Fail with a readable message if the ESM2 runtime packages are missing."""
    if ML_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing required ML dependency. Activate the environment that has "
            "torch, fair-esm, and scipy installed before running the walk. "
            f"Original import error: {ML_IMPORT_ERROR}"
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basic setup helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device_arg


def get_coupling_sample_pairs(length: int) -> int:
    """Pick a practical number of residue pairs to test for coupling."""
    return min(2000, max(500, length * 3))


def get_protein_config(length: int, args: argparse.Namespace) -> WalkConfig:
    """Set the baseline/adaptive PRCS-only walk parameters from protein length."""

    # k_single is the number of positions changed in a normal guided proposal.
    k_single = max(1, round(length / 50))

    # Coupled mutations use groups. Bigger proteins get slightly larger groups.
    group_size_min = max(3, round(length / 80))
    group_size_max = max(5, round(length / 40))

    # Make enough groups to cover a useful fraction of the sequence.
    avg_group = (group_size_min + group_size_max) / 2
    n_groups = max(15, int(0.7 * length / avg_group))

    # Larger proteins use coupled proposals more often, but never above 80%.
    p_coupled = min(0.8, 0.3 + length / 1000)

    # Baseline step rule: max(2000, 10 * length). No extra big-protein multiplier here.
    max_steps = max(args.min_steps, args.steps_per_residue * length)

    # Temperature decays from 0.3 to about 0.005 across the full run.
    cooling_rate = (0.005 / INITIAL_TEMP) ** (1.0 / max_steps)

    return WalkConfig(
        k_single=k_single,
        group_size_min=group_size_min,
        group_size_max=group_size_max,
        n_groups=n_groups,
        p_coupled=p_coupled,
        max_steps=max_steps,
        cooling_rate=cooling_rate,
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FASTA and sequence helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_fasta_one(path: Path) -> tuple[str, str]:
    """Read one FASTA file and return (header_name, sequence)."""
    lines = path.read_text().strip().splitlines()
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"{path} is not a valid FASTA file.")
    name = lines[0].lstrip(">").strip().split()[0]
    seq = "".join(line.strip() for line in lines[1:] if not line.startswith(">"))
    seq = seq.replace(" ", "").upper()
    if not seq:
        raise ValueError(f"{path} has no sequence.")
    bad = sorted(set(seq) - set(AMINO_ACIDS))
    if bad:
        raise ValueError(f"{path} contains unsupported residues: {''.join(bad)}")
    return name, seq


def write_sequence_fasta(path: Path, name: str, seq: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as handle:
        handle.write(f">{name}\n")
        for start in range(0, len(seq), 60):
            handle.write(seq[start : start + 60] + "\n")


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Hamming distance requires equal-length sequences.")
    return sum(x != y for x, y in zip(a, b))


def mutate_positions(seq: str, positions: list[int], aas: list[str]) -> str:
    chars = list(seq)
    for pos, aa in zip(positions, aas):
        chars[pos] = aa
    return "".join(chars)


def random_sequence(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(AMINO_ACIDS) for _ in range(length))


def find_fasta_files(fasta_dir: Path) -> list[Path]:
    """Find FASTA files in a folder without requiring exact file extensions."""
    files = sorted(fasta_dir.glob("*.fasta"))
    files.extend(sorted(fasta_dir.glob("*.fa")))
    return files


def load_proteins(args: argparse.Namespace) -> list[dict]:
    """Load target proteins from the command-line FASTA folder."""
    if not args.fasta_dir.exists():
        raise FileNotFoundError(f"FASTA directory not found: {args.fasta_dir}")
    fasta_files = find_fasta_files(args.fasta_dir)
    if not fasta_files:
        raise FileNotFoundError(f"No .fasta or .fa files found in {args.fasta_dir}")

    proteins = []
    for path in fasta_files:
        _, seq = read_fasta_one(path)
        proteins.append({"name": path.stem, "seq": seq, "length": len(seq), "path": path})

    if args.protein_filter.strip():
        requested = {name.strip().lower() for name in args.protein_filter.split(",") if name.strip()}
        proteins = [protein for protein in proteins if protein["name"].lower() in requested]
        if not proteins:
            raise ValueError(f"--protein-filter matched no FASTA files: {args.protein_filter}")
    return proteins


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ESM2 model and scoring helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_esm2(device: str, model_name: str):
    if model_name != "esm2_t33_650M_UR50D":
        raise ValueError("Only --model esm2_t33_650M_UR50D is supported in this script.")
    print("Loading ESM2...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(torch.device(device))
    batch_converter = alphabet.get_batch_converter()
    print(f"  Model loaded on {device}")
    return model, alphabet, batch_converter


@no_grad()
def get_per_residue_embedding(seq, model, alphabet, batch_converter, device):
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    out = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
    return out["representations"][REPR_LAYER][0, 1:-1, :]


@no_grad()
def get_per_residue_embeddings_batch(
    seqs, model, alphabet, batch_converter, device, max_batch=None
):
    if not seqs:
        return []
    if max_batch is None:
        length = len(seqs[0])
        if length < 150:
            max_batch = 20
        elif length < 300:
            max_batch = 12
        elif length < 500:
            max_batch = 8
        else:
            max_batch = 4

    all_embeddings = []
    for start in range(0, len(seqs), max_batch):
        batch_seqs = seqs[start : start + max_batch]
        data = [(f"s{i}", seq) for i, seq in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
        reps = out["representations"][REPR_LAYER]
        for i in range(len(batch_seqs)):
            all_embeddings.append(reps[i, 1:-1, :].clone())
        del reps, out, tokens
    return all_embeddings


def score_prcs(emb_candidate, emb_wt) -> tuple[float, np.ndarray]:
    per_pos = F.cosine_similarity(emb_candidate, emb_wt, dim=1)
    return per_pos.mean().item(), per_pos.cpu().numpy()


def score_epistatic(emb_candidate, emb_wt, groups: list[list[int]]) -> float:
    if not groups:
        return 0.0
    cosines = []
    for group in groups:
        for i, j in combinations(group, 2):
            diff_candidate = emb_candidate[i] - emb_candidate[j]
            diff_wt = emb_wt[i] - emb_wt[j]
            cos = F.cosine_similarity(
                diff_candidate.unsqueeze(0), diff_wt.unsqueeze(0)
            ).item()
            cosines.append(cos)
    return float(np.mean(cosines)) if cosines else 0.0


def compute_combined_score(emb_candidate, emb_wt, groups, lam):
    prcs, per_pos = score_prcs(emb_candidate, emb_wt)
    epi = score_epistatic(emb_candidate, emb_wt, groups) if lam > 0.01 else 0.0
    return prcs + lam * epi, prcs, epi, per_pos


@no_grad()
def score_true_pll(seq, model, alphabet, batch_converter, device, pos_batch_size=16) -> float:
    """Masked pseudo-log-likelihood diagnostic. Only logged, not optimized."""
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    length = len(seq)
    log_probs = []
    for start in range(0, length, pos_batch_size):
        end = min(start + pos_batch_size, length)
        batch_tokens = tokens.repeat(end - start, 1)
        for idx, pos in enumerate(range(start, end)):
            batch_tokens[idx, pos + 1] = alphabet.mask_idx
        out = model(batch_tokens, repr_layers=[], return_contacts=False)
        logits = out["logits"]
        for idx, pos in enumerate(range(start, end)):
            log_p = torch.log_softmax(logits[idx, pos + 1], dim=-1)
            true_aa = tokens[0, pos + 1].item()
            log_probs.append(log_p[true_aa].item())
    return float(np.mean(log_probs))


@no_grad()
def compute_single_masked_marginals(seq, model, alphabet, batch_converter, device, positions):
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    aa_indices = [alphabet.get_idx(aa) for aa in AMINO_ACIDS]
    marginals = {}
    for pos in positions:
        masked = tokens.clone()
        masked[0, pos + 1] = alphabet.mask_idx
        out = model(masked, repr_layers=[], return_contacts=False)
        logits = out["logits"][0, pos + 1]
        probs = torch.softmax(logits, dim=-1)
        marginals[pos] = probs[aa_indices].cpu().numpy()
    return marginals


@no_grad()
def compute_double_masked_marginals(seq, model, alphabet, batch_converter, device, pos_i, pos_j):
    data = [("seq", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    aa_indices = [alphabet.get_idx(aa) for aa in AMINO_ACIDS]
    masked = tokens.clone()
    masked[0, pos_i + 1] = alphabet.mask_idx
    masked[0, pos_j + 1] = alphabet.mask_idx
    out = model(masked, repr_layers=[], return_contacts=False)

    logits_i = out["logits"][0, pos_i + 1]
    logits_j = out["logits"][0, pos_j + 1]
    probs_i = torch.softmax(logits_i, dim=-1)
    probs_j = torch.softmax(logits_j, dim=-1)
    return probs_i[aa_indices].cpu().numpy(), probs_j[aa_indices].cpu().numpy()


@no_grad()
def compute_epistatic_coupling_map(
    seq, model, alphabet, batch_converter, device, n_sample_pairs: int, rng: random.Random
):
    length = len(seq)
    all_pairs = list(combinations(range(length), 2))
    sampled_pairs = rng.sample(all_pairs, min(n_sample_pairs, len(all_pairs)))

    unique_positions = sorted({pos for pair in sampled_pairs for pos in pair})
    single_marginals = compute_single_masked_marginals(
        seq, model, alphabet, batch_converter, device, unique_positions
    )

    coupling_scores = []
    for i, j in sampled_pairs:
        double_i, double_j = compute_double_masked_marginals(
            seq, model, alphabet, batch_converter, device, i, j
        )
        jsd_i = jensenshannon(single_marginals[i], double_i)
        jsd_j = jensenshannon(single_marginals[j], double_j)
        coupling = (jsd_i + jsd_j) / 2
        if not np.isnan(coupling):
            coupling_scores.append((i, j, float(coupling)))

    coupling_scores.sort(key=lambda item: item[2], reverse=True)
    return coupling_scores


def build_coupling_groups(
    coupling_scores,
    length: int,
    n_groups: int,
    group_size_min: int,
    group_size_max: int,
    threshold: float,
    rng: random.Random,
):
    """Turn strong pair couplings into groups that can be mutated together."""
    effective_threshold = threshold
    if coupling_scores:
        all_strengths = [score for _, _, score in coupling_scores]
        n_above = sum(score >= threshold for score in all_strengths)
        frac_above = n_above / len(all_strengths)
        if frac_above < COUPLING_MIN_FRAC:
            sorted_strengths = sorted(all_strengths, reverse=True)
            idx = int(len(sorted_strengths) * (1 - COUPLING_MIN_FRAC))
            effective_threshold = sorted_strengths[min(idx, len(sorted_strengths) - 1)]
            effective_threshold = max(effective_threshold, 0.01)

    adjacency = {i: [] for i in range(length)}
    for i, j, strength in coupling_scores:
        if strength >= effective_threshold:
            adjacency[i].append((j, strength))
            adjacency[j].append((i, strength))

    coupling_count = {i: len(neighbors) for i, neighbors in adjacency.items()}
    sorted_positions = sorted(range(length), key=lambda pos: coupling_count[pos], reverse=True)

    used = set()
    groups = []
    for seed_pos in sorted_positions:
        if seed_pos in used or len(groups) >= n_groups:
            continue
        if coupling_count[seed_pos] == 0:
            break

        group = [seed_pos]
        used.add(seed_pos)
        neighbors = sorted(adjacency[seed_pos], key=lambda item: item[1], reverse=True)
        for nb, _ in neighbors:
            if nb not in used and len(group) < group_size_max:
                group.append(nb)
                used.add(nb)

        while len(group) < group_size_min:
            remaining = [pos for pos in range(length) if pos not in used]
            if not remaining:
                break
            pos = rng.choice(remaining)
            group.append(pos)
            used.add(pos)

        if len(group) >= group_size_min:
            groups.append(group)

    while len(groups) < n_groups:
        remaining = [pos for pos in range(length) if pos not in used]
        if len(remaining) < group_size_min:
            break
        group = remaining[:group_size_min]
        used.update(group)
        groups.append(group)

    return groups, effective_threshold


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Mutation proposal helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def propose_single(seq: str, k_mutations: int, rng: random.Random):
    """Mutate k random positions to random non-current amino acids."""
    length = len(seq)
    positions = rng.sample(range(length), min(k_mutations, length))
    aas = []
    for pos in positions:
        candidates = [aa for aa in AMINO_ACIDS if aa != seq[pos]]
        aas.append(rng.choice(candidates))
    return mutate_positions(seq, positions, aas), positions


def propose_single_guided(
    seq: str,
    k_mutations: int,
    per_pos_prcs: np.ndarray,
    rng: random.Random,
    p_guided: float = 0.85,
):
    """Mutate mostly weak PRCS positions, while keeping some random exploration."""
    length = len(seq)
    k_mutations = min(k_mutations, length)

    # The weak pool is the worst-scoring 30% of residue positions by PRCS.
    n_worst = max(k_mutations, int(0.3 * length))
    worst_indices = np.argsort(per_pos_prcs)[:n_worst].tolist()

    positions = []
    used = set()
    for _ in range(k_mutations):
        available_all = [pos for pos in range(length) if pos not in used]

        # Most of the time, pick from weak PRCS positions.
        # Sometimes pick from anywhere, so the walk can still explore.
        if rng.random() < p_guided:
            available_worst = [pos for pos in worst_indices if pos not in used]
            pos = rng.choice(available_worst or available_all)
        else:
            pos = rng.choice(available_all)
        positions.append(pos)
        used.add(pos)

    aas = []
    for pos in positions:
        candidates = [aa for aa in AMINO_ACIDS if aa != seq[pos]]
        aas.append(rng.choice(candidates))
    return mutate_positions(seq, positions, aas), positions


def propose_coupled(seq: str, groups: list[list[int]], rng: random.Random):
    """Mutate every residue in one coupling-derived group."""
    if not groups:
        return propose_single(seq, 1, rng)
    positions = list(rng.choice(groups))
    aas = []
    for pos in positions:
        candidates = [aa for aa in AMINO_ACIDS if aa != seq[pos]]
        aas.append(rng.choice(candidates))
    return mutate_positions(seq, positions, aas), positions


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Adaptive score-weight and mutation-size logic
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def compute_adaptive_lambda_max(coupling_scores, threshold: float):
    """Set the largest epistasis weight allowed for the current coupling map."""
    above = [score for _, _, score in coupling_scores if score >= threshold]
    if not above:
        return LAMBDA_MIN, 0.0
    mean_coupling = float(np.mean(above))
    lambda_max = float(np.clip(mean_coupling * LAMBDA_COUPLING_SCALE, LAMBDA_MIN, LAMBDA_CAP))
    return lambda_max, mean_coupling


def compute_lambda(current_prcs: float, start_prcs: float, lambda_max: float):
    """Ramp lambda up as PRCS improves, instead of trusting epistasis immediately."""
    gap = 1.0 - start_prcs
    if gap <= 0:
        return lambda_max
    progress = (current_prcs - start_prcs) / gap
    ramp = min(1.0, max(0.0, progress / LAMBDA_RAMP_FRAC))
    return lambda_max * ramp


def choose_adaptive_refinement(
    trajectory,
    current_prcs: float,
    start_prcs: float,
    k_single: int,
    p_coupled: float,
    args: argparse.Namespace,
):
    """Choose how careful mutation proposals should be right now."""

    # Convert current PRCS into progress from the random start toward the WT target.
    # Example: start_prcs=0.50 and current_prcs=0.75 means progress=0.50.
    gap = max(1e-6, 1.0 - start_prcs)
    progress = max(0.0, min(1.0, (current_prcs - start_prcs) / gap))

    # adaptive_window means: look at the last N steps and ask whether PRCS is still
    # improving. Default N=400, so the script waits for enough history before declaring a plateau.
    plateau = False
    if len(trajectory) >= args.adaptive_window * 2:

        # earlier = everything before the recent window.
        earlier = trajectory[: -args.adaptive_window]

        # recent = the last adaptive_window steps.
        recent = trajectory[-args.adaptive_window :]

        # Compare best PRCS from the earlier part and the recent part.
        earlier_best = max(row["prcs"] for row in earlier)
        recent_best = max(row["prcs"] for row in recent)

        # If PRCS improved by less than adaptive_prcs_eps, call it a plateau.
        plateau = (recent_best - earlier_best) < args.adaptive_prcs_eps

    # Stage 0 is normal baseline-style movement.
    stage = 0

    # Stage 1: sequence has made moderate PRCS progress, so reduce big jumps.
    if progress >= args.adaptive_stage1:
        stage = 1

    # Stage 2: sequence is closer to WT-like, so use even smaller moves.
    if progress >= args.adaptive_stage2:
        stage = 2

    # Stage 3: sequence is strongly WT-like, so use almost single-residue refinement.
    if progress >= args.adaptive_stage3:
        stage = 3

    # If PRCS stalls after some progress, make the walk one stage more precise.
    if plateau and progress >= args.adaptive_plateau:
        stage = min(3, max(1, stage + 1))

    # Final stage: mostly single-residue cleanup, very few coupled moves.
    if stage >= 3:
        return 1, min(p_coupled, 0.10), stage, progress, plateau

    # Middle stage: small proposals, fewer coupled-group mutations.
    if stage == 2:
        return min(k_single, 2), min(p_coupled, 0.25), stage, progress, plateau

    # Early refinement stage: cap normal mutation size and coupled probability.
    if stage == 1:
        return min(k_single, 4), min(p_coupled, 0.40), stage, progress, plateau

    # No adaptive refinement yet.
    return k_single, p_coupled, stage, progress, plateau


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Output and checkpoint helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def save_checkpoint(path: Path, state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(state, handle)


def load_checkpoint(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def checkpoint_path(outdir: Path, protein_name: str, run_id: int) -> Path:
    return outdir / protein_name / f"checkpoint_run{run_id}.pkl"


def write_rows(path: Path, rows: list[dict], fieldnames: list[str], gzip_output: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_output else open
    mode = "wt" if gzip_output else "w"
    with opener(path, mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def infer_fieldnames(rows: list[dict], preferred: Iterable[str]) -> list[str]:
    seen = []
    for field in preferred:
        if any(field in row for row in rows):
            seen.append(field)
    extra = sorted({key for row in rows for key in row.keys()} - set(seen))
    return seen + extra


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# One-run initialization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def initialize_run(
    wt,
    config: WalkConfig,
    emb_wt,
    model,
    alphabet,
    batch_converter,
    device,
    rng,
    protein_name,
    run_id,
):
    length = len(wt)
    print(
        f"  [Run {run_id}] L={length}  k_single={config.k_single}  "
        f"group_size={config.group_size_min}-{config.group_size_max}  "
        f"p_coupled={config.p_coupled:.2f}  steps={config.max_steps}"
    )

    current_seq = random_sequence(length, rng)
    start_seq = current_seq
    current_emb = get_per_residue_embedding(current_seq, model, alphabet, batch_converter, device)
    start_prcs, current_per_pos_prcs = score_prcs(current_emb, emb_wt)
    current_hamming = hamming(current_seq, wt)
    current_combined = start_prcs

    print(f"  [Run {run_id}] Start PRCS={start_prcs:.4f}  Hamming={current_hamming}")

    n_sample_pairs = get_coupling_sample_pairs(length)
    print(
        f"  [Run {run_id}] Computing initial coupling map "
        f"(double-masking, {n_sample_pairs} pairs)..."
    )
    coupling_scores = compute_epistatic_coupling_map(
        current_seq, model, alphabet, batch_converter, device, n_sample_pairs, rng
    )
    n_above = sum(1 for _, _, score in coupling_scores if score >= COUPLING_THRESHOLD)
    print(
        f"  [Run {run_id}] {n_above}/{len(coupling_scores)} pairs above "
        f"coupling threshold {COUPLING_THRESHOLD}"
    )
    groups, effective_threshold = build_coupling_groups(
        coupling_scores,
        length,
        config.n_groups,
        config.group_size_min,
        config.group_size_max,
        COUPLING_THRESHOLD,
        rng,
    )
    group_sizes = [len(group) for group in groups]
    print(
        f"  [Run {run_id}] {len(groups)} coupled groups, "
        f"sizes: {group_sizes[:10]}{'...' if len(group_sizes) > 10 else ''}"
    )
    if effective_threshold != COUPLING_THRESHOLD:
        print(
            f"  [Run {run_id}] Threshold adapted: "
            f"{COUPLING_THRESHOLD} -> {effective_threshold:.4f}"
        )
    lambda_max, mean_coupling = compute_adaptive_lambda_max(coupling_scores, effective_threshold)
    print(
        f"  [Run {run_id}] Adaptive lambda_max={lambda_max:.3f} "
        f"(mean coupling={mean_coupling:.4f})"
    )

    return {
        "step_start": 0,
        "current_seq": current_seq,
        "current_emb": current_emb,
        "current_prcs": start_prcs,
        "current_combined": current_combined,
        "current_hamming": current_hamming,
        "current_per_pos_prcs": current_per_pos_prcs,
        "start_seq": start_seq,
        "start_prcs": start_prcs,
        "start_hamming": current_hamming,
        "best_prcs_seq": current_seq,
        "best_prcs": start_prcs,
        "hamming_at_best_prcs": current_hamming,
        "best_combined": current_combined,
        "min_hamming_seq": current_seq,
        "min_hamming": current_hamming,
        "prcs_at_min_hamming": start_prcs,
        "temperature": INITIAL_TEMP,
        "lambda_max": lambda_max,
        "effective_threshold": effective_threshold,
        "groups": groups,
        "n_sample_pairs": n_sample_pairs,
        "trajectory": [],
        "n_accepted_single": 0,
        "n_proposed_single": 0,
        "n_accepted_coupled": 0,
        "n_proposed_coupled": 0,
    }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main evolution-walk loop
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_walk(
    wt: str,
    protein_name: str,
    run_id: int,
    config: WalkConfig,
    emb_wt,
    model,
    alphabet,
    batch_converter,
    device: str,
    rng: random.Random,
    outdir: Path,
    args: argparse.Namespace,
):
    length = len(wt)
    ckpt = checkpoint_path(outdir, protein_name, run_id)
    if ckpt.exists() and not args.overwrite:
        print(f"  [Run {run_id}] Resuming from checkpoint: {ckpt}")
        state = load_checkpoint(ckpt)
        rng.setstate(state["rng_state"])
        state["current_emb"] = get_per_residue_embedding(
            state["current_seq"], model, alphabet, batch_converter, device
        )
        _, state["current_per_pos_prcs"] = score_prcs(state["current_emb"], emb_wt)
    else:
        state = initialize_run(
            wt,
            config,
            emb_wt,
            model,
            alphabet,
            batch_converter,
            device,
            rng,
            protein_name,
            run_id,
        )

    for step in range(state["step_start"], config.max_steps):
        lam = compute_lambda(state["current_prcs"], state["start_prcs"], state["lambda_max"])

        if step > 0 and step % args.coupling_recompute_every == 0:
            print(f"  [Run {run_id}] Step {step}: recomputing coupling map")
            coupling_scores = compute_epistatic_coupling_map(
                state["current_seq"],
                model,
                alphabet,
                batch_converter,
                device,
                state["n_sample_pairs"],
                rng,
            )
            state["groups"], state["effective_threshold"] = build_coupling_groups(
                coupling_scores,
                length,
                config.n_groups,
                config.group_size_min,
                config.group_size_max,
                COUPLING_THRESHOLD,
                rng,
            )
            state["lambda_max"], _ = compute_adaptive_lambda_max(
                coupling_scores, state["effective_threshold"]
            )

        (
            effective_k_single,
            effective_p_coupled,
            adaptive_stage,
            adaptive_progress,
            adaptive_plateau,
        ) = choose_adaptive_refinement(
            state["trajectory"],
            state["current_prcs"],
            state["start_prcs"],
            config.k_single,
            config.p_coupled,
            args,
        )

        cand_seqs = []
        cand_is_coupled = []
        for _ in range(args.proposals_per_step):
            if rng.random() < effective_p_coupled and state["groups"]:
                cand_seq, _ = propose_coupled(state["current_seq"], state["groups"], rng)
                is_coupled = True
            else:
                cand_seq, _ = propose_single_guided(
                    state["current_seq"],
                    effective_k_single,
                    state["current_per_pos_prcs"],
                    rng,
                    p_guided=0.85,
                )
                is_coupled = False
            cand_seqs.append(cand_seq)
            cand_is_coupled.append(is_coupled)

        cand_embeddings = get_per_residue_embeddings_batch(
            cand_seqs, model, alphabet, batch_converter, device
        )

        best_idx = -1
        best_cand_score = -float("inf")
        best_cand_prcs = -float("inf")
        best_cand_epi = 0.0
        best_cand_per_pos = None

        for idx, cand_emb in enumerate(cand_embeddings):
            combined, prcs, epi, per_pos = compute_combined_score(
                cand_emb, emb_wt, state["groups"], lam
            )
            if combined > best_cand_score:
                best_idx = idx
                best_cand_score = combined
                best_cand_prcs = prcs
                best_cand_epi = epi
                best_cand_per_pos = per_pos

            if cand_is_coupled[idx]:
                state["n_proposed_coupled"] += 1
            else:
                state["n_proposed_single"] += 1

        delta = best_cand_score - state["current_combined"]
        if delta > 0:
            accepted = True
        elif state["temperature"] > 1e-10:
            accepted = rng.random() < math.exp(delta / state["temperature"])
        else:
            accepted = False

        if accepted:
            state["current_seq"] = cand_seqs[best_idx]
            state["current_emb"] = cand_embeddings[best_idx]
            state["current_prcs"] = best_cand_prcs
            state["current_combined"] = best_cand_score
            state["current_hamming"] = hamming(state["current_seq"], wt)
            state["current_per_pos_prcs"] = best_cand_per_pos

            if cand_is_coupled[best_idx]:
                state["n_accepted_coupled"] += 1
            else:
                state["n_accepted_single"] += 1

            if state["current_prcs"] > state["best_prcs"]:
                state["best_prcs"] = state["current_prcs"]
                state["best_prcs_seq"] = state["current_seq"]
                state["hamming_at_best_prcs"] = state["current_hamming"]
                state["best_combined"] = state["current_combined"]

            if state["current_hamming"] < state["min_hamming"]:
                state["min_hamming"] = state["current_hamming"]
                state["min_hamming_seq"] = state["current_seq"]
                state["prcs_at_min_hamming"] = state["current_prcs"]

        state["temperature"] *= config.cooling_rate

        row = {
            "step": step,
            "prcs": state["current_prcs"],
            "epi": best_cand_epi,
            "combined": state["current_combined"],
            "hamming": state["current_hamming"],
            "best_prcs": state["best_prcs"],
            "hamming_at_best_prcs": state["hamming_at_best_prcs"],
            "min_hamming": state["min_hamming"],
            "prcs_at_min_hamming": state["prcs_at_min_hamming"],
            "lambda": lam,
            "lambda_max": state["lambda_max"],
            "temperature": state["temperature"],
            "accepted": accepted,
            "coupled": cand_is_coupled[best_idx],
            "k_single_effective": effective_k_single,
            "effective_p_coupled": effective_p_coupled,
            "adaptive_stage": adaptive_stage,
            "adaptive_progress": adaptive_progress,
            "adaptive_plateau": adaptive_plateau,
            "delta": delta,
        }
        if not args.no_sequence_trajectory:
            row["sequence"] = state["current_seq"]
        state["trajectory"].append(row)

        if step > 0 and step % args.log_every == 0:
            print(
                f"  [Run {run_id}] Step {step}  "
                f"PRCS={state['current_prcs']:.4f}  "
                f"Epi={best_cand_epi:.4f}  "
                f"Combined={state['current_combined']:.4f}  "
                f"Hamming={state['current_hamming']}  "
                f"MinH={state['min_hamming']}  "
                f"k={effective_k_single}  p_c={effective_p_coupled:.2f}  "
                f"stage={adaptive_stage}  prog={adaptive_progress:.2f}  "
                f"lambda={lam:.3f}(max={state['lambda_max']:.3f})  "
                f"T={state['temperature']:.4f}"
            )

        if step > 0 and args.pll_diagnostic_every > 0 and step % args.pll_diagnostic_every == 0:
            pll = score_true_pll(state["current_seq"], model, alphabet, batch_converter, device)
            print(f"  [Run {run_id}] Step {step} PLL diagnostic: {pll:.4f}")
            state["trajectory"][-1]["pll"] = pll

        if step > 0 and args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            checkpoint_state = {key: value for key, value in state.items() if key != "current_emb"}
            checkpoint_state["step_start"] = step + 1
            checkpoint_state["rng_state"] = rng.getstate()
            save_checkpoint(ckpt, checkpoint_state)
            print(f"  [Run {run_id}] Checkpoint saved at step {step}")

    acc_single = (
        state["n_accepted_single"] / state["n_proposed_single"] * 100
        if state["n_proposed_single"]
        else 0.0
    )
    acc_coupled = (
        state["n_accepted_coupled"] / state["n_proposed_coupled"] * 100
        if state["n_proposed_coupled"]
        else 0.0
    )
    final_hamming = hamming(state["current_seq"], wt)

    print(f"  [Run {run_id}] Finished after {config.max_steps} steps")
    print(f"    Final PRCS={state['current_prcs']:.4f}  Best PRCS={state['best_prcs']:.4f}")
    print(
        f"    Final Hamming={final_hamming}  Min Hamming={state['min_hamming']}  "
        f"Hamming at best PRCS={state['hamming_at_best_prcs']}"
    )
    print(f"    Accept rate: single={acc_single:.1f}%  coupled={acc_coupled:.1f}%")

    summary = {
        "protein": protein_name,
        "length": length,
        "run": run_id,
        "start_prcs": state["start_prcs"],
        "start_hamming": state["start_hamming"],
        "best_prcs": state["best_prcs"],
        "hamming_at_best_prcs": state["hamming_at_best_prcs"],
        "best_combined": state["best_combined"],
        "min_hamming": state["min_hamming"],
        "prcs_at_min_hamming": state["prcs_at_min_hamming"],
        "final_prcs": state["current_prcs"],
        "final_hamming": final_hamming,
        "final_combined": state["current_combined"],
        "final_lambda": compute_lambda(state["current_prcs"], state["start_prcs"], state["lambda_max"]),
        "final_lambda_max": state["lambda_max"],
        "accept_rate_single": acc_single,
        "accept_rate_coupled": acc_coupled,
        "steps": config.max_steps,
        "k_single": config.k_single,
        "group_size_min": config.group_size_min,
        "group_size_max": config.group_size_max,
        "p_coupled": config.p_coupled,
        "n_groups": len(state["groups"]),
    }

    if ckpt.exists():
        ckpt.unlink()
        print("    Checkpoint removed (run completed)")

    return {
        "trajectory": state["trajectory"],
        "summary": summary,
        "start_seq": state["start_seq"],
        "final_seq": state["current_seq"],
        "best_prcs_seq": state["best_prcs_seq"],
        "min_hamming_seq": state["min_hamming_seq"],
    }


def save_run_outputs(protein_outdir: Path, run_id: int, result: dict, args: argparse.Namespace):
    """Write the trajectories, summaries, and key FASTA sequences for one run."""
    trajectory = result["trajectory"]
    preferred_fields = [
        "step",
        "prcs",
        "epi",
        "combined",
        "hamming",
        "best_prcs",
        "hamming_at_best_prcs",
        "min_hamming",
        "prcs_at_min_hamming",
        "lambda",
        "lambda_max",
        "temperature",
        "accepted",
        "coupled",
        "k_single_effective",
        "effective_p_coupled",
        "adaptive_stage",
        "adaptive_progress",
        "adaptive_plateau",
        "delta",
        "pll",
        "sequence",
    ]
    fields = infer_fieldnames(trajectory, preferred_fields)
    if not args.no_sequence_trajectory and "sequence" in fields:
        write_rows(
            protein_outdir / f"sequence_trajectory_run{run_id}.csv.gz",
            trajectory,
            fields,
            gzip_output=True,
        )
        fields_without_sequence = [field for field in fields if field != "sequence"]
        write_rows(
            protein_outdir / f"trajectory_run{run_id}.csv",
            trajectory,
            fields_without_sequence,
            gzip_output=False,
        )
    else:
        write_rows(protein_outdir / f"trajectory_run{run_id}.csv", trajectory, fields)

    summary_fields = list(result["summary"].keys())
    write_rows(protein_outdir / f"summary_run{run_id}.csv", [result["summary"]], summary_fields)

    write_sequence_fasta(
        protein_outdir / f"start_sequence_run{run_id}.fasta",
        f"{protein_outdir.name}_run{run_id}_start_H{result['summary']['start_hamming']}",
        result["start_seq"],
    )
    write_sequence_fasta(
        protein_outdir / f"final_sequence_run{run_id}.fasta",
        f"{protein_outdir.name}_run{run_id}_final_H{result['summary']['final_hamming']}",
        result["final_seq"],
    )
    write_sequence_fasta(
        protein_outdir / f"best_prcs_sequence_run{run_id}.fasta",
        (
            f"{protein_outdir.name}_run{run_id}_best_prcs_"
            f"H{result['summary']['hamming_at_best_prcs']}"
        ),
        result["best_prcs_seq"],
    )
    write_sequence_fasta(
        protein_outdir / f"min_hamming_sequence_run{run_id}.fasta",
        f"{protein_outdir.name}_run{run_id}_min_hamming_H{result['summary']['min_hamming']}",
        result["min_hamming_seq"],
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Console summary and entry point
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def print_header(proteins: list[dict], args: argparse.Namespace, device: str):
    print("=" * 72)
    print("  ESM2 ADAPTIVE EVOLUTION WALK")
    print(f"  FASTA directory: {args.fasta_dir}")
    print(f"  Output directory: {args.outdir}")
    if args.protein_filter.strip():
        print(f"  Protein filter: {args.protein_filter}")
    print(f"  Device: {device}")
    print(f"  Runs per protein: {args.runs_per_protein}")
    print(f"  Steps: max({args.min_steps}, {args.steps_per_residue} * length)")
    print(f"  Proposals per step: {args.proposals_per_step}")
    print(
        "  Adaptive stages: "
        f"{args.adaptive_stage1:.2f}, {args.adaptive_stage2:.2f}, "
        f"{args.adaptive_stage3:.2f}; window={args.adaptive_window}, "
        f"eps={args.adaptive_prcs_eps}"
    )
    print("  Score: PRCS + lambda * epistasis")
    print("=" * 72)
    for protein in proteins:
        cfg = get_protein_config(protein["length"], args)
        print(
            f"  {protein['name']:35s} {protein['length']:4d} aa  "
            f"k={cfg.k_single}  steps={cfg.max_steps}  "
            f"group={cfg.group_size_min}-{cfg.group_size_max}  "
            f"p_coupled={cfg.p_coupled:.2f}  "
            f"pairs={get_coupling_sample_pairs(protein['length'])}"
        )
    print()


def main() -> int:
    args = parse_args()
    ensure_ml_dependencies()
    device = resolve_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    proteins = load_proteins(args)
    print_header(proteins, args, device)

    model, alphabet, batch_converter = load_esm2(device, args.model)
    all_summaries = []

    for run_id in range(args.runs_per_protein):
        print()
        print("=" * 72)
        print(f"  TRANCHE {run_id + 1}/{args.runs_per_protein}")
        print("=" * 72)

        for protein_index, protein in enumerate(proteins):
            protein_outdir = args.outdir / protein["name"]
            protein_outdir.mkdir(parents=True, exist_ok=True)
            trajectory_path = protein_outdir / f"trajectory_run{run_id}.csv"
            if trajectory_path.exists() and not args.overwrite:
                print(f"\n  [{protein['name']}] run {run_id} already exists; skipping.")
                print("  Use --overwrite to rerun it.")
                continue

            print()
            print("#" * 72)
            print(
                f"  PROTEIN {protein_index + 1}/{len(proteins)}: "
                f"{protein['name']} [run {run_id}]"
            )
            print("#" * 72)

            wt = protein["seq"]
            cfg = get_protein_config(protein["length"], args)
            print("  Computing WT reference embedding...")
            emb_wt = get_per_residue_embedding(wt, model, alphabet, batch_converter, device)

            if run_id == 0:
                test_rng = random.Random(0)
                test_seq = random_sequence(protein["length"], test_rng)
                test_emb = get_per_residue_embedding(test_seq, model, alphabet, batch_converter, device)
                rand_prcs, _ = score_prcs(test_emb, emb_wt)
                print("  WT PRCS: 1.000000")
                print(
                    f"  Random PRCS: {rand_prcs:.4f}  "
                    f"Random Hamming: {hamming(test_seq, wt)}"
                )

            run_rng = random.Random(args.seed + protein_index * 100 + run_id)
            start_time = time.time()
            result = run_walk(
                wt,
                protein["name"],
                run_id,
                cfg,
                emb_wt,
                model,
                alphabet,
                batch_converter,
                device,
                run_rng,
                args.outdir,
                args,
            )
            elapsed_min = (time.time() - start_time) / 60
            save_run_outputs(protein_outdir, run_id, result, args)
            all_summaries.append(result["summary"])

            print(
                f"  Saved {protein['name']} run {run_id} in {elapsed_min:.1f} min: "
                f"best PRCS={result['summary']['best_prcs']:.4f}, "
                f"min Hamming={result['summary']['min_hamming']}, "
                f"final Hamming={result['summary']['final_hamming']}"
            )

    if all_summaries:
        summary_fields = list(all_summaries[0].keys())
        write_rows(args.outdir / "cross_protein_summary.csv", all_summaries, summary_fields)

        print()
        print("=" * 72)
        print("  SUMMARY")
        print("=" * 72)
        print(f"  {'Protein':<28s} {'L':>4s} {'Best PRCS':>9s} {'H@Best':>7s} {'MinH':>6s} {'FinalH':>7s}")
        for row in all_summaries:
            print(
                f"  {row['protein']:<28s} {row['length']:4d} "
                f"{row['best_prcs']:9.4f} {row['hamming_at_best_prcs']:7d} "
                f"{row['min_hamming']:6d} {row['final_hamming']:7d}"
            )

    print(f"\nDone. Results saved to {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
