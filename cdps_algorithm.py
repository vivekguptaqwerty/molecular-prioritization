"""
CDPS — Candidate Drug Prioritisation Scheme
=============================================

Implements the algorithm described in Chapter 5 of the thesis (Section 5.3,
Equations 19-21): a multi-criteria ranking of candidate drugs that combines

    1. predicted sensitivity        (e.g. GNAE-Drugnet's predicted IC50 / LN(IC50))
    2. predicted binding affinity   (e.g. ProtLigGNN's predicted pKd / ΔG)
    3. molecular novelty            (Tanimoto distance to a known compound library)

into a single composite score, with the three weights learned (not assigned
by hand) via a pairwise ranking loss.

This script is self-contained and only needs numbers you already have from
your trained models — it does not retrain GNAE-Drugnet or ProtLigGNN itself.
Run it wherever those predictions and the RDKit/PyTorch environment live.

--------------------------------------------------------------------------
EXPECTED INPUT
--------------------------------------------------------------------------
A CSV with one row per candidate (drug, or drug-cell-line pair) and columns:

    candidate_id       - any unique identifier (string)
    smiles             - SMILES string of the candidate (for novelty)
    predicted_ic50      - GNAE-Drugnet's predicted value (lower = more sensitive,
                           e.g. predicted LN(IC50) — sign convention handled below)
    predicted_affinity  - ProtLigGNN's predicted value (higher = stronger binder,
                           e.g. predicted pKd)
    observed_response   - the ground-truth quantity used to define "correct"
                           pairwise ordering for training/evaluating the ranker
                           (e.g. measured LN(IC50), or measured pIC50/pKd).
                           Lower observed_response is assumed = more sensitive/
                           stronger, matching predicted_ic50's convention; flip
                           --observed-higher-is-better if that is not the case.

A second CSV (or the same one, plus a --library-smiles file) lists the SMILES
of the KNOWN compound library against which novelty is computed.

If you don't have real data yet, run with --demo to see the whole pipeline
execute end-to-end on synthetic data, including the Table-5.2-style output.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    pip install rdkit torch scipy pandas scikit-learn --break-system-packages

    # demo on synthetic data (sanity check the pipeline runs)
    python cdps_algorithm.py --demo

    # real run
    python cdps_algorithm.py \
        --candidates candidates.csv \
        --library library_smiles.csv \
        --test-frac 0.3 \
        --seed 0 \
        --out cdps_results.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

try:
    import torch
except ImportError:
    sys.exit("This script requires PyTorch. Install with: pip install torch --break-system-packages")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    HAVE_RDKIT = True
except ImportError:
    HAVE_RDKIT = False


# ---------------------------------------------------------------------------
# 1. Novelty — Equation 19:  T(a,b) = |a∩b| / |a∪b|,  N_i = 1 - max_{b in L} T(a_i, b)
# ---------------------------------------------------------------------------

def compute_novelty(candidate_smiles, library_smiles, radius=2, n_bits=2048):
    """
    Tanimoto-based novelty score for each candidate against a known library.
    Returns an array of novelty scores in [0, 1]; 1 = maximally novel
    (no similar compound in the library), 0 = identical to a library member.
    """
    if not HAVE_RDKIT:
        raise RuntimeError(
            "RDKit is required for novelty computation. "
            "Install with: pip install rdkit --break-system-packages\n"
            "(Or pre-compute a 'novelty' column yourself and pass --novelty-column.)"
        )

    def fp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

    lib_fps = [f for f in (fp(s) for s in library_smiles) if f is not None]
    if not lib_fps:
        raise ValueError("No valid SMILES parsed from the library file.")

    novelty = np.zeros(len(candidate_smiles))
    for i, smi in enumerate(candidate_smiles):
        cfp = fp(smi)
        if cfp is None:
            novelty[i] = np.nan  # unparseable candidate; caller should drop/flag it
            continue
        sims = DataStructs.BulkTanimotoSimilarity(cfp, lib_fps)
        novelty[i] = 1.0 - max(sims)
    return novelty


# ---------------------------------------------------------------------------
# 2. Rank normalisation — put each signal on a common [0, 1] scale by rank
# ---------------------------------------------------------------------------

def rank_normalise(values, higher_is_better=True):
    """
    Converts a raw signal into a rank-normalised score in [0, 1], where
    1.0 = the best candidate on this signal and 0.0 = the worst.
    Ties are averaged (standard competition-safe rank handling).
    """
    values = np.asarray(values, dtype=float)
    ranks = rankdata(values, method="average")  # 1 = lowest raw value
    if not higher_is_better:
        ranks = (len(values) + 1) - ranks
    return (ranks - 1) / (len(values) - 1) if len(values) > 1 else np.zeros_like(ranks)


# ---------------------------------------------------------------------------
# 3. CDPS weight learning — Equations 20-21
#    S_i = w1*r(IC50_i) + w2*r(ΔĜ_i) + w3*r(N_i),  sum(w) = 1, w >= 0
#    L_rank = sum_{(i,j) in P} log(1 + exp(-sigma * (S_i - S_j)))
# ---------------------------------------------------------------------------

class CDPS(torch.nn.Module):
    """
    Learns the three CDPS signal weights by minimising a pairwise ranking
    loss. Weights are parameterised through a softmax so they are
    automatically non-negative and sum to one, matching Equation 20's
    constraint without needing a separate projection step.
    """

    def __init__(self, n_signals=3, sigma=1.0, learn_sigma=False):
        super().__init__()
        self.raw_weights = torch.nn.Parameter(torch.zeros(n_signals))
        self.log_sigma = torch.nn.Parameter(
            torch.log(torch.tensor(float(sigma))), requires_grad=learn_sigma
        )

    @property
    def weights(self):
        return torch.softmax(self.raw_weights, dim=0)

    def score(self, signal_matrix):
        # signal_matrix: (n_candidates, n_signals), each column already rank-normalised
        return signal_matrix @ self.weights

    def pairwise_loss(self, signal_matrix, pair_i, pair_j):
        """
        pair_i, pair_j: index tensors such that candidate i is known to
        correctly outrank candidate j (i.e. observed_response says i is
        better than j). This is Equation 21.
        """
        s = self.score(signal_matrix)
        sigma = torch.exp(self.log_sigma)
        diff = s[pair_i] - s[pair_j]
        return torch.nn.functional.softplus(-sigma * diff).mean()


def build_training_pairs(observed_response, higher_is_better=False, max_pairs=200_000, seed=0):
    """
    Builds the set P of ordered pairs (i, j) such that i is known to be the
    better candidate, drawn from the observed ground-truth response.
    Equal-response pairs are excluded (no correct ordering exists for them).
    For large candidate sets, pairs are subsampled to max_pairs for
    tractability; all pairs are used otherwise.
    """
    rng = np.random.default_rng(seed)
    n = len(observed_response)
    order = np.argsort(observed_response)  # ascending: index 0 = lowest value
    if higher_is_better:
        order = order[::-1]  # index 0 = best

    all_pairs = [(order[a], order[b]) for a in range(n) for b in range(a + 1, n)]
    if len(all_pairs) > max_pairs:
        idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        all_pairs = [all_pairs[k] for k in idx]

    pair_i = torch.tensor([p[0] for p in all_pairs], dtype=torch.long)
    pair_j = torch.tensor([p[1] for p in all_pairs], dtype=torch.long)
    return pair_i, pair_j


def train_cdps(signal_matrix, observed_response, higher_is_better=False,
               epochs=500, lr=0.05, sigma=1.0, seed=0, verbose=True):
    torch.manual_seed(seed)
    model = CDPS(n_signals=signal_matrix.shape[1], sigma=sigma)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    pair_i, pair_j = build_training_pairs(observed_response, higher_is_better, seed=seed)

    signal_t = torch.tensor(signal_matrix, dtype=torch.float32)
    for epoch in range(epochs):
        optimiser.zero_grad()
        loss = model.pairwise_loss(signal_t, pair_i, pair_j)
        loss.backward()
        optimiser.step()
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:4d}  pairwise ranking loss = {loss.item():.4f}")

    learned_weights = model.weights.detach().numpy()
    return learned_weights


# ---------------------------------------------------------------------------
# 4. Evaluation — reproduces the structure of Table 5.2
# ---------------------------------------------------------------------------

def evaluate_ordering(score, observed_response, higher_is_better=False):
    """Spearman rank correlation between a candidate score and ground truth."""
    truth = -np.asarray(observed_response) if not higher_is_better else np.asarray(observed_response)
    rho, _ = spearmanr(score, truth)
    return rho


def run_ablation_table(r_ic50, r_affinity, r_novelty, observed_response,
                        higher_is_better=False, learned_weights=None, equal_weights=None):
    """Prints and returns a Table-5.2-style comparison of signal combinations."""
    rows = []

    def add(name, score, condition):
        rho = evaluate_ordering(score, observed_response, higher_is_better)
        rows.append({"signal_combination": name, "spearman": round(rho, 3), "condition": condition})

    add("Predicted sensitivity only", r_ic50, "Single signal")
    add("Predicted binding affinity only", r_affinity, "Single signal")
    add("Molecular novelty only", r_novelty, "Single signal")
    add("Sensitivity and affinity", 0.5 * r_ic50 + 0.5 * r_affinity, "Two signals")
    add("Sensitivity and novelty", 0.5 * r_ic50 + 0.5 * r_novelty, "Two signals")

    if equal_weights is None:
        equal_weights = np.array([1 / 3, 1 / 3, 1 / 3])
    s_equal = equal_weights[0] * r_ic50 + equal_weights[1] * r_affinity + equal_weights[2] * r_novelty
    add("All three, equal weights", s_equal, "Heuristic weighting")

    if learned_weights is not None:
        s_learned = (learned_weights[0] * r_ic50 + learned_weights[1] * r_affinity
                     + learned_weights[2] * r_novelty)
        add("All three, learned weights (CDPS)", s_learned,
            f"Proposed algorithm (w={np.round(learned_weights, 3).tolist()})")

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 5. CLI / orchestration
# ---------------------------------------------------------------------------

def make_demo_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    true_latent = rng.normal(size=n)
    predicted_ic50 = -true_latent + rng.normal(scale=0.6, size=n)       # correlated signal 1
    predicted_affinity = 0.7 * true_latent + rng.normal(scale=0.8, size=n)  # correlated signal 2
    novelty = rng.uniform(0, 1, size=n)                                  # near-independent signal
    observed_response = -true_latent + rng.normal(scale=0.3, size=n)     # ground truth
    df = pd.DataFrame({
        "candidate_id": [f"cand_{i}" for i in range(n)],
        "predicted_ic50": predicted_ic50,
        "predicted_affinity": predicted_affinity,
        "novelty": novelty,
        "observed_response": observed_response,
    })
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", type=str, help="CSV of candidates (see module docstring for columns)")
    p.add_argument("--library", type=str, help="CSV with a 'smiles' column: the known compound library")
    p.add_argument("--novelty-column", type=str, default=None,
                   help="If novelty is already computed, name of that column; skips RDKit")
    p.add_argument("--observed-higher-is-better", action="store_true",
                   help="Set if higher observed_response = better (default: lower is better, e.g. IC50/LN(IC50))")
    p.add_argument("--test-frac", type=float, default=0.3, help="Held-out fraction for evaluation")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="cdps_results.csv")
    p.add_argument("--demo", action="store_true", help="Run on synthetic data instead of real files")
    args = p.parse_args()

    if args.demo:
        print("Running CDPS pipeline on synthetic demo data (no RDKit / real files needed)...\n")
        df = make_demo_data(seed=args.seed)
    else:
        if not args.candidates:
            sys.exit("Provide --candidates <file.csv> or use --demo.")
        df = pd.read_csv(args.candidates)
        required = {"candidate_id", "predicted_ic50", "predicted_affinity", "observed_response"}
        missing = required - set(df.columns)
        if missing:
            sys.exit(f"Missing required column(s) in --candidates: {missing}")

        if args.novelty_column:
            df["novelty"] = df[args.novelty_column]
        elif "smiles" in df.columns and args.library:
            lib_df = pd.read_csv(args.library)
            if "smiles" not in lib_df.columns:
                sys.exit("--library CSV must have a 'smiles' column.")
            print("Computing Tanimoto novelty against the known library...")
            df["novelty"] = compute_novelty(df["smiles"].tolist(), lib_df["smiles"].tolist())
            n_bad = df["novelty"].isna().sum()
            if n_bad:
                print(f"  warning: {n_bad} candidate SMILES could not be parsed; dropping them.")
                df = df.dropna(subset=["novelty"]).reset_index(drop=True)
        else:
            sys.exit("Need either --novelty-column, or a 'smiles' column plus --library.")

    higher_is_better = args.observed_higher_is_better

    # --- split ---
    rng = np.random.default_rng(args.seed)
    n = len(df)
    idx = rng.permutation(n)
    n_test = int(round(n * args.test_frac))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    train_df, test_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    print(f"\n{len(train_df)} training candidates, {len(test_df)} held-out test candidates.")

    # --- rank-normalise signals (fit on train, apply same convention to test) ---
    def signals_for(d):
        r_ic50 = rank_normalise(d["predicted_ic50"].values, higher_is_better=False)      # lower IC50 = better
        r_aff = rank_normalise(d["predicted_affinity"].values, higher_is_better=True)    # higher affinity = better
        r_nov = rank_normalise(d["novelty"].values, higher_is_better=True)               # higher novelty = better
        return np.stack([r_ic50, r_aff, r_nov], axis=1), r_ic50, r_aff, r_nov

    train_matrix, *_ = signals_for(train_df)
    test_matrix, test_ic50, test_aff, test_nov = signals_for(test_df)

    # --- learn weights on TRAIN only ---
    print("\nLearning CDPS weights on the training set (pairwise ranking loss)...")
    learned_weights = train_cdps(
        train_matrix, train_df["observed_response"].values,
        higher_is_better=higher_is_better, epochs=args.epochs, lr=args.lr,
        sigma=args.sigma, seed=args.seed,
    )
    print(f"\nLearned weights  w_ic50={learned_weights[0]:.3f}  "
          f"w_affinity={learned_weights[1]:.3f}  w_novelty={learned_weights[2]:.3f}")

    # --- evaluate on held-out TEST set (this is the honest, non-circular number) ---
    print("\n=== Held-out test-set ranking quality (Table 5.2 equivalent) ===")
    results_df = run_ablation_table(
        test_ic50, test_aff, test_nov, test_df["observed_response"].values,
        higher_is_better=higher_is_better, learned_weights=learned_weights,
    )

    # --- composite score + ranked output for every test candidate ---
    s_learned = (learned_weights[0] * test_ic50 + learned_weights[1] * test_aff
                 + learned_weights[2] * test_nov)
    out_df = test_df.copy()
    out_df["cdps_score"] = s_learned
    out_df = out_df.sort_values("cdps_score", ascending=False).reset_index(drop=True)
    out_df["cdps_rank"] = np.arange(1, len(out_df) + 1)
    out_df.to_csv(args.out, index=False)
    print(f"\nRanked candidates written to {args.out}")
    print(f"Learned weights: {dict(zip(['w_ic50', 'w_affinity', 'w_novelty'], np.round(learned_weights, 4)))}")

    results_df.to_csv(args.out.replace(".csv", "_ablation_table.csv"), index=False)
    print(f"Ablation table written to {args.out.replace('.csv', '_ablation_table.csv')}")


if __name__ == "__main__":
    main()
