# Final Results Summary

## Project Scope

ProtLigGNN predicts protein-ligand binding affinity from paired ligand molecular graphs and protein pocket graphs. The GenAI extension adds a closed-loop workflow: generated ligands are chemically validated with RDKit, prioritized by ProtLigGNN affinity and GAPS, then secondarily checked with AutoDock Vina docking.

Novelty statement: the central novelty is the closed-loop GenAI + ProtLigGNN affinity-guided ligand prioritization framework, not RDKit or AutoDock Vina themselves.

RDKit and Vina are validation and filtering tools in this project. They support chemical validity and secondary docking checks, but they are not claimed as the novel contribution.

## ProtLigGNN Best Model Metrics

- Best run: no_crossgraph_5000_e40
- PCC: 0.6570
- Spearman: 0.6540
- RMSE: 1.4068
- MAE: 1.0550

## Ablation Comparison

|run_name|best_epoch|pcc|spearman|rmse|mae|
|---|---|---|---|---|---|
|no_crossgraph_5000_e40|37|0.6570|0.6540|1.4068|1.0550|
|crossgraph_5000_e40|32|0.5944|0.5614|1.4793|1.2023|

## Earlier Report-Asset Experiment Table

|run_name|best_epoch|test_pcc|test_spearman|test_rmse|test_mae|
|---|---|---|---|---|---|
|crossgraph_2000|12|0.4993|0.4408|1.5642|1.189|
|crossgraph_500|7|0.3554|0.3029|1.7626|1.33|
|crossgraph_5000_e40|32|0.5944|0.5614|1.4793|1.2023|
|no_crossgraph_2000|15|0.511|0.4719|1.5912|1.2418|
|no_crossgraph_500|10|0.3392|0.2688|1.7778|1.2851|
|no_crossgraph_5000_e40|37|0.657|0.654|1.4068|1.055|

## Top Known Binder Results

|rank|pdb_id|predicted_affinity|true_affinity|absolute_error|
|---|---|---|---|---|
|1|4kb9|10.739169|11.000000|0.260831|
|2|3o9d|10.321081|10.721247|0.400166|
|3|4hdb|10.118496|8.050610|2.067886|
|4|4i8w|10.117595|10.896196|0.778602|
|5|5kqy|9.857601|9.292430|0.565171|
|6|4yo8|9.764246|10.522879|0.758633|
|7|3o9d|9.740625|10.721247|0.980621|
|8|4ll3|9.669128|11.568636|1.899508|
|9|4uxl|9.577358|10.602060|1.024702|
|10|4flh|9.525223|9.000000|0.525223|

## GenAI Generation Metrics

- generated_count: 10000
- valid_count: 6072
- unique_valid_count: 4097
- novel_unique_count: 3845
- validity_percentage: 60.72
- uniqueness_percentage: 67.47364953886694
- novelty_percentage: 93.84915792042958

## GAPS Explanation

GAPS means Generative Affinity Prioritization Score. In this project it combines normalized ProtLigGNN affinity, novelty, normalized QED, and diversity to rank generated molecules before docking validation.

## Top Generated Candidates

|smiles|predicted_affinity|gaps|qed|lipinski_violations|
|---|---|---|---|---|
|N#Cc1ccc(CNC(=O)c2ccc(Cl)c(Cl)c2)cn1|6.8907694816589355|0.8695540971027537|0.9467335565659457|0|
|N#Cc1cccc(NC(=O)OCc2ccccc2)c1|7.348996639251709|0.8595435656664115|0.9110660939697823|0|
|C[C@H]1CCC(O)=CC=C1CNc1ncnc2cccnc12|7.224691390991211|0.8555782232067641|0.9039838966959223|0|
|COc1ccc(NC2=NC(c3ccn(C)n3)=CC=CC=C2)cc1|6.534128189086914|0.8546889567941968|0.9452035454175607|0|
|O=C(COc1cccc(Cl)c1)Nc1ccccc1F|6.817463397979736|0.8340183203237191|0.93015837334839|0|
|CCC(=O)Nc1cc(Nc2ccc(C(F)(F)F)cn2)ccn1|7.22393274307251|0.833627991230567|0.9054256879991691|0|
|O=S(=O)(NCc1nccs1)c1ccc(C(F)(F)F)cc1|6.531422138214111|0.8261698759832218|0.9412428707861615|0|
|COc1ccc(F)cc1NC(=O)c1ccc(S(N)(=O)=O)cc1|7.505073070526123|0.8198744865639728|0.8945483874004212|0|
|FC(F)(F)c1ccccc1Nc1ccnc(N2CCOCC2)n1|6.3271098136901855|0.8108679257610268|0.9401863053715935|0|
|COc1cccc(C(=O)Nc2ccc(C#N)cc2Cl)c1|6.482480525970459|0.8092596589899802|0.939484226814529|0|

## Vina Docking Validation Results

### Known Top 10

|rank|pdb_id|predicted_affinity|true_affinity|abs_error|vina_score|
|---|---|---|---|---|---|
|1|4kb9|10.739169|11.000000|0.260831|-5.354|
|2|3o9d|10.321081|10.721247|0.400166|-7.451|
|3|4hdb|10.118496|8.050610|2.067886|-5.777|
|4|4i8w|10.117595|10.896196|0.778602|-10.04|
|5|5kqy|9.857601|9.292430|0.565171|-5.31|
|6|4yo8|9.764246|10.522879|0.758633|-9.381|
|7|3o9d|9.740625|10.721247|0.980621|-10.66|
|8|4ll3|9.669128|11.568636|1.899508|-4.417|
|9|4uxl|9.577358|10.602060|1.024702|-12.17|
|10|4flh|9.525223|9.000000|0.525223|-9.729|

### Generated Top 10

|rank|smiles|predicted_affinity|gaps_score|qed|lipinski_violations|vina_score|
|---|---|---|---|---|---|---|
|1|N#Cc1ccc(CNC(=O)c2ccc(Cl)c(Cl)c2)cn1|6.8907694816589355|0.8695540971027537|0.9467335565659457|0|-6.383|
|2|N#Cc1cccc(NC(=O)OCc2ccccc2)c1|7.348996639251709|0.8595435656664115|0.9110660939697823|0|-6.484|
|3|C[C@H]1CCC(O)=CC=C1CNc1ncnc2cccnc12|7.224691390991211|0.8555782232067641|0.9039838966959223|0|-6.88|
|4|COc1ccc(NC2=NC(c3ccn(C)n3)=CC=CC=C2)cc1|6.534128189086914|0.8546889567941968|0.9452035454175607|0|-6.776|
|5|O=C(COc1cccc(Cl)c1)Nc1ccccc1F|6.817463397979736|0.8340183203237191|0.93015837334839|0|-7.389|
|6|CCC(=O)Nc1cc(Nc2ccc(C(F)(F)F)cn2)ccn1|7.22393274307251|0.833627991230567|0.9054256879991691|0|-7.003|
|7|O=S(=O)(NCc1nccs1)c1ccc(C(F)(F)F)cc1|6.531422138214111|0.8261698759832218|0.9412428707861615|0|-6.612|
|8|COc1ccc(F)cc1NC(=O)c1ccc(S(N)(=O)=O)cc1|7.505073070526123|0.8198744865639728|0.8945483874004212|0|-6.535|
|9|FC(F)(F)c1ccccc1Nc1ccnc(N2CCOCC2)n1|6.3271098136901855|0.8108679257610268|0.9401863053715935|0|-7.632|
|10|COc1cccc(C(=O)Nc2ccc(C#N)cc2Cl)c1|6.482480525970459|0.8092596589899802|0.939484226814529|0|-6.911|

## Best Generated Molecule Result

- SMILES: FC(F)(F)c1ccccc1Nc1ccnc(N2CCOCC2)n1
- ProtLigGNN predicted affinity: 6.3271098136901855
- GAPS: 0.8108679257610268
- QED: 0.9401863053715935
- Lipinski violations: 0
- Vina score: -7.632

## Output Evidence

- ProtLigGNN comparison: outputs/comparison_all_models_5000_e40.txt
- GenAI quality summary: outputs/genai/generated_molecule_quality_summary.json
- Generated candidate ranking: outputs/genai/top10_generated_candidates_by_gaps.csv
- Docking comparison: outputs/vina_validation/final_docking_comparison.md
