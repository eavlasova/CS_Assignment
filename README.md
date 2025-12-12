# Product Duplicate Detection Pipeline

This repository implements a Python pipeline to detect duplicate products across shops using MinHash and LSH for candidate pair generation and XGBoost for supervised duplicate classification.

## Features

- **LSH-based candidate generation** for scalable approximate duplicate detection.
- **XGBoost refinement** for supervised verification of candidate pairs.
- **Bootstrap evaluation** for robust performance metrics.
- Reports **Pair Quality (PQ), Pair Completeness (PC), and F1 score**.
- Supports configurable **LSH thresholds, permutations, and band parameters**.

## Pipeline Overview

1. **Data Preparation**
   - Load product data
   - Extract model world from title
   - Build binary vectors for model words

2. **LSH Candidate Generation**
   - Compute MinHash signatures for each product.
   - Build LSH buckets and generate candidate pairs
   - Apply constraints: exclude same shop, incompatible brands
   - Evaluate LSH-only performance (PQ, PC, F1)

3. **Bootstrap Sampling**
   - Stratified bootstrapping ensures balanced cluster representation.
   - Repeat LSH candidate generation and evaluation across multiple bootstraps

4. **Feature Extraction for XGBoost**
   - Compute features for each LSH candidate pair:
     - Jaccard similarity of model words
     - Brand match
     - KVP-based similarity
   - Label pairs as duplicates (`1`) or not (`0`)

5. **XGBoost Training and Evaluation**
   - Train on bootstrap candidate pairs.
   - Evaluate on candidate pairs from held-out evaluation set
   - Report PQ, PC, F1 for both LSH and XGBoost
   - Aggregate results across bootstraps.

## Requirements

- Python >= 3.8
- numpy
- pandas
- xgboost
- scikit-learn
- matplotlib (optional, for additional plots)
- itertools, collections, math, random, time (standard Python libraries)
- typing, re (standard Python libraries)
