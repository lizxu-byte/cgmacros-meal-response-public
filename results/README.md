# results/

Produced by `python scripts/run_all.py`. Everything here is **aggregated**:
by health-state group (healthy / prediabetes / type 2 diabetes) or over the
whole cohort. No meal-level rows, per-participant rows, raw readings or meal
photos are written.

These tables and figures are derived from CGMacros v1.0.0 (PhysioNet,
doi:10.13026/3z8q-x658) and are therefore shared under the dataset's licence,
**CC BY-NC-SA 4.0** (attribution, non-commercial, share-alike). The code in
this repository is MIT-licensed; the MIT licence does not apply to this folder.

| File | Content |
|---|---|
| `table_cohort_by_group.csv` | n, median age / BMI / HbA1c / fasting glucose / insulin, % male per group |
| `table_cgm_agreement.csv` | Libre Pro vs Dexcom G6 agreement (bias, Bland-Altman LoA, MARD, r), all and by group |
| `table_cgm_lag_scan.csv` | median within-participant r and bias with Libre shifted by -20..+20 min |
| `table_glycemic_by_group_{dexcom,libre}.csv` | glycemic metrics by group with Kruskal-Wallis, eta^2_H, pairwise Holm p, Cliff's delta |
| `table_meal_exclusions.csv` | meal records, episodes and sequential exclusion counts per device and group |
| `table_overlap_exclusion_by_group.csv` | episodes excluded by the 120-min rule by group (pooled and per-participant share) |
| `table_cv_threshold_by_group.csv` | participants with CV < 36 % (Danne et al. 2017 stability threshold) by group and device |
| `table_photos.csv` | number of photo files and meals with photos (not modelled) |
| `table_meal_response_by_group.csv` | per-participant median iAUC / peak rise / time to peak compared across groups, overall and by carbohydrate bin |
| `table_mixed_model_coefficients.csv` | fixed effects of the mixed models (primary, sensitivity, replication) |
| `table_mixed_model_carb_slopes.csv` | carbohydrate slope within each group |
| `table_mixed_model_predicted_iauc.csv` | back-transformed predicted iAUC by group at 30 / 60 / 90 g carbohydrate |
| `table_prediction_ablation.csv` | modality ablation: pooled and per-participant r, R^2, MAE for LOSO and temporal settings |
| `table_group_classification*.csv` | participant-level group classification (wearable / CGM features) and confusion matrices (group counts) |
| `table_wearable_by_group.csv` | wearable participant features by group |
| `summary.json` | key numbers used in the top-level README |
| `figures/` | Bland-Altman density, postprandial curves, mixed-model coefficients, ablation charts |
