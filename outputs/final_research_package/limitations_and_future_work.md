# Limitations and Future Work

## Limitations

- ProtLigGNN performance is evaluated computationally; experimental binding validation is not included.
- The SMILES LSTM is lightweight and character-level, so chemical syntax and scaffold quality depend heavily on RDKit filtering after generation.
- AutoDock Vina is used as secondary computational validation, not as ground truth.
- RDKit descriptors and validity checks are filtering tools and are not the novelty of the project.
- Docking boxes are computational approximations and may not capture all induced-fit or solvent effects.

## Future Work

- Train stronger molecular generators such as graph-based or transformer-based models.
- Add protein-conditioned generation rather than target-agnostic SMILES generation.
- Add more rigorous docking box selection around known binding pockets.
- Validate top generated candidates with molecular dynamics or experimental assays.
- Expand the closed-loop GenAI + ProtLigGNN system with active learning from scored candidates.
- Extend beyond single protein-ligand binding affinity (pKd) to drug-response prediction: a cell-line-conditioned GNN over drug and cell-line features predicting IC50, alongside a reinforcement-learning-based molecule generator (optimizing directly against the affinity/response predictor rather than supervised SMILES sampling) and a combined drug-pair sensitivity (CDPS) scoring scheme. This was part of the original project scope but was not implemented in the current codebase; this paper reports the completed ProtLigGNN + GenAI + docking pipeline only.
