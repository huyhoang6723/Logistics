# Logistics

This repository contains the reproducibility materials for the study:

**An Interpretable Two-Part Statistical Framework for Semicontinuous Delivery-Burden Modeling in Fulfillment Systems**

The repository includes the primary DataCo analysis, the independent Olist framework replication, the corresponding data archives, and the generated results.

## Repository contents

- `code.py`  
  Primary analysis for the DataCo Smart Supply Chain dataset. It reconstructs order-level data, builds the cancellation-aware semicontinuous burden outcome, fits the occurrence and severity components, evaluates expected burden, runs robustness analyses, and generates the reported tables and figures.

- `data.zip`  
  Data archive for the primary DataCo analysis.

- `results.zip`  
  Outputs from the primary DataCo analysis.

- `external.py`  
  Independent external framework replication using the Brazilian E-Commerce Public Dataset by Olist. The script reads the settings produced by the primary DataCo analysis, reconstructs an analogous burden outcome on Olist, refits the same analytical framework on Olist, and evaluates external performance and robustness.

- `dataexternal.zip`  
  Olist data archive used for the external framework replication.

- `external.zip`  
  Outputs from the Olist external framework replication.

- `LICENSE`  
  Repository license.

## Data sources

### DataCo Smart Supply Chain

The primary analysis uses the DataCo Smart Supply Chain for Big Data Analysis dataset.

Source: Mendeley Data  
DOI: `10.17632/8gx2fvg2k6.5`

The analysis is performed at the order level after reconstruction from the original product-line records.

### Olist Brazilian E-Commerce

The external framework replication uses the Brazilian E-Commerce Public Dataset by Olist, distributed through Kaggle.

The Olist archive contains the original linked CSV tables used by `external.py`, including the orders, order items, customers, products, sellers, and geolocation files where available.

## Python environment

Python 3.12 is recommended.

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Primary DataCo analysis

Place the DataCo CSV in the same folder as `code.py`, or provide the path explicitly.

The primary script can be run as:

```bash
python code.py DataCoSupplyChainDataset.csv
```

If no input path is provided, the script searches the current directory for a file matching the DataCo dataset naming pattern.

Example:

```bash
python code.py
```

The script creates an output directory named from the input dataset, for example:

```text
DataCoSupplyChainDataset_revision_analysis_output/
```

It also generates the main workbook:

```text
two_part_delivery_glm_revision_results.xlsx
```

and a packaged ZIP archive of the analysis outputs.

A custom output directory may be supplied with:

```bash
python code.py DataCoSupplyChainDataset.csv --output-dir my_results
```

## Olist external framework replication

The external analysis requires:

1. the Olist data archive, and
2. the primary DataCo results workbook `two_part_delivery_glm_revision_results.xlsx`.

Place these files in the same directory as `external.py`.

The simplest command is:

```bash
python external.py
```

By default, the script searches for:

```text
two_part_delivery_glm_revision_results.xlsx
archive.zip
```

or another ZIP archive containing the required Olist CSV files.

If the files have different names, provide them explicitly:

```bash
python external.py --results two_part_delivery_glm_revision_results.xlsx --archive dataexternal.zip
```

The default external output directory is:

```text
external_olist_results/
```

The main external workbook is:

```text
olist_external_validation_results.xlsx
```

A custom output directory can be specified with:

```bash
python external.py --output-dir my_external_results
```

The bootstrap replication count can also be changed, for example:

```bash
python external.py --bootstrap 1000
```

## Reproducibility design

The primary DataCo workflow uses a chronological 60/20/20 train-validation-test split. Preprocessing rules, reconstruction parameters, model settings, and other quantities that must be estimated are learned without using the final test set.

The external Olist analysis is an **independent framework replication**, not a direct transport of the fitted DataCo coefficients. The analytical structure and relevant settings are carried forward, while the occurrence and severity models are re-estimated on Olist because the two datasets do not share an identical predictor space.

The external script reads the prior DataCo results workbook so that settings such as the split proportions, occurrence-model regularization, conformal level, top-k evaluation capacities, and primary cancellation-penalty rule remain aligned with the primary analysis.

## Main analytical components

The workflow includes:

- order-level reconstruction;
- point-in-time predictor auditing;
- chronological validation;
- train-derived burden reconstruction;
- two-part occurrence-severity modeling;
- L2-regularized logistic occurrence prediction;
- log-linear positive-severity modeling with Duan retransformation;
- expected-burden scoring;
- calibration assessment;
- conformal severity intervals;
- top-k burden capture;
- operational triage;
- same-data ranking benchmarks;
- penalty, grace, and outcome-definition sensitivity analyses;
- feature-retention and post-QR audit;
- independent Olist framework replication.

## Important interpretation note

The proposed framework is intended as an interpretable and auditable statistical baseline for burden-aware prioritization. The Olist analysis evaluates whether the analytical framework can be replicated in an independent operational dataset; it should not be interpreted as direct transport of the DataCo fitted coefficients or as evidence of universal superiority over alternative ranking scores.

## Reproducing the paper results

For the primary results:

```bash
python code.py DataCoSupplyChainDataset.csv
```

For the external replication, copy or retain the resulting `two_part_delivery_glm_revision_results.xlsx` beside `external.py` and run:

```bash
python external.py --results two_part_delivery_glm_revision_results.xlsx --archive dataexternal.zip
```

The generated workbooks, CSV outputs, figures, environment information, file hashes, and manifests provide the audit trail used to reproduce the manuscript results.
