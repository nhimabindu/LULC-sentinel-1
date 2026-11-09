# Sentinel-1 Multi-City Land Use and Land Cover (LULC) Mapping

## Project Overview

This project develops a monthly land use and land cover (LULC) mapping workflow using **Sentinel-1 SAR data**. It produces monthly maps for 12 cities for January–December 2025, analyses changes in class coverage over time, and evaluates the model against reference and comparison datasets.

The workflow uses Sentinel-1 VV and VH backscatter data and a semantic segmentation model to classify six LULC classes. Precomputed maps, statistics, model checkpoints and evaluation results are also used by the Streamlit dashboard.

## Study Areas

The project covers 12 cities, grouped by their role in the experiments:

| Group | Cities |
|---|---|
| Training | Bengaluru, Berlin, Cairo, Hamburg, Munich, New York, São Paulo |
| Validation | Freiburg, Singapore |
| Test | Heidelberg, Nairobi, Sydney |

## LULC Classes

The model uses the following class IDs:

| Class ID | Land-cover class |
|---:|---|
| 0 | Tree cover |
| 1 | Grassland |
| 2 | Cropland |
| 3 | Built-up |
| 4 | Bare / sparse vegetation |
| 5 | Permanent water |
| 255 | NoData / ignored pixels |

## Data Sources

- **Sentinel-1:** VV and VH SAR observations from the Sentinel-1 GRD collection. Monthly composites are used as model inputs.
- **ESA WorldCover 2021:** Used to derive the reference labels for model development and evaluation. These labels are a proxy reference and do not represent verified 2025 ground truth.
- **Dynamic World:** Used as an independent, Sentinel-2-derived comparison product for cross-sensor evaluation. It is not treated as ground truth.

## Methodology

1. **Data preparation:** Organise monthly Sentinel-1 VV/VH composites and the WorldCover-derived class labels for each study area.
2. **Preprocessing:** Check spatial alignment and valid data, apply the project's training-derived SAR normalisation statistics, and prepare model inputs.
3. **Patch generation:** Extract 64 × 64 pixel patches using a stride of 32. Patches are filtered using SAR and label-validity thresholds; ignored pixels use class ID `255`.
4. **Model training:** Train semantic segmentation models using the training and validation splits.
5. **Prediction:** Apply the selected model to monthly city composites and save georeferenced LULC prediction rasters.
6. **Change analysis:** Calculate class percentages and month-to-month changes from the saved predictions.
7. **Evaluation:** Assess model performance against the held-out proxy labels and compare predictions with Dynamic World where valid spatial coverage is available.

## Models and Evaluation

Two segmentation architectures were explored: U-Net and a lightweight FCN-8s-inspired model. The FCN model was selected for the final monthly mapping workflow.

Reported test-set results for the selected FCN model:

| Metric | Result |
|---|---:|
| Overall accuracy | 64.46% |
| Mean intersection over union (mIoU) | 29.65% |
| Mean F1-score | 40.04% |

These metrics measure agreement with the available WorldCover-derived proxy labels. Class imbalance is substantial, and performance varies considerably between classes. In particular, rare classes such as cropland and bare/sparse vegetation are difficult to predict reliably.

Dynamic World results should be interpreted as **cross-sensor/product agreement**, not as accuracy against ground truth. Coverage varies by city and month, so agreement scores should always be considered together with the valid-pixel coverage.

## Project Structure

```text
LULC_Sentinel1_Project/
│
├── README.md
├── configs/
│   ├── cities.yaml
│   ├── classes.yaml
│   └── project.yaml
│
├── data/
│   ├── raw/
│   │   └── sentinel1/
│   ├── labels/
│   └── processed/
│       ├── sentinel1/
│       └── patches/
│           ├── train/
│           ├── val/
│           └── test/
│
├── models/
│   ├── fcn8s_best.pt
│   └── unet_best.pt
│
├── notebooks/
│   └── Data.ipynb
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── training/
│   ├── inference/
│   └── evaluation/
│
├── dashboard/
│   ├── app.py
│   ├── app.ipynb
│   └── backups/
│
├── outputs/
│   ├── data_audit/
│   ├── evaluation/
│   ├── independent_validation/
│   ├── monthly_maps/
│   ├── statistics/
│   └── visualizations/
│
├── logs/
└── archive/
    ├── notebook_checkpoints/
    └── old_dashboard_backups/
```

## Main Outputs

- **Monthly maps:** Georeferenced LULC prediction rasters for the city-month combinations.
- **Statistics:** Monthly class percentages, month-to-month class changes, and pixel transition summaries.
- **Data audits:** Input-data checks, patch manifests, class distributions and normalisation information.
- **Model evaluation:** Test metrics, confusion matrices, class-wise metrics and training histories.
- **Independent comparison:** Dynamic World availability, coverage and cross-sensor comparison metrics.
- **Visualisations:** Monthly map panels, class distributions, change summaries and evaluation charts.

## Dashboard

The Streamlit dashboard presents the precomputed project outputs, including monthly maps, class distributions, change analysis, model evaluation and Dynamic World comparison.

The map view visualises saved predictions; it does not perform new model inference for arbitrary coordinates.

## Reproducing the Workflow

The main data preparation notebook is `notebooks/Data.ipynb`. The trained checkpoints and generated outputs are stored separately in `models/` and `outputs/`.

To inspect the existing results, open the saved notebooks, model evaluation files, maps and dashboard assets. **You do not need to retrain the models or regenerate the monthly maps just to review the existing results.**

Re-running the full workflow is only necessary when intentionally reproducing or changing the experiments.

The exact Python package versions and launch commands should be recorded in a project environment file (for example, `requirements.txt`) once the working environment has been finalised.

## Important Limitations

- WorldCover 2021 labels are used as proxy labels for the 2025 Sentinel-1 inputs; they are not contemporaneous ground truth.
- Strong class imbalance affects model performance, especially for rare land-cover classes.
- Dynamic World is derived from Sentinel-2 and is an independent comparison product, not ground truth.
- Dynamic World coverage differs across city-month combinations. Low-coverage comparisons should not be interpreted as reliable city-wide agreement.
- Results should be interpreted with these data and evaluation limitations in mind.