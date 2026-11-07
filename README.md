# DTI - Drug-Target Interaction Platform

## Overview

A professional-grade cheminformatics and machine learning platform for drug discovery.
Built with free public APIs (ChEMBL, PubChem) and Python.

**Repository:** https://github.com/mahimmazidul/dti  
**Author:** mahimmazidul  
**License:** MIT  

## Features

### Core Modules (Original)
1. **ChEMBL Query** (`src/chembl_query.py`) - Search ChEMBL bioactivities
2. **PubChem Query** (`src/pubchem_query.py`) - Search PubChem compounds
3. **Analyzer** (`src/analyzer.py`) - Main logic engine
4. **OpenFDA Safety Module** (`src/openfda_query.py`) - Drug interactions, warnings, adverse events
5. **Network Graph Visualizer** (`src/visualizer.py`) - Drug-target interaction networks
6. **Drug Repurposing Scanner** (`src/repurposing.py`) - Clinical stage compound scanning
7. **CLI** (`cli.py`) - Command-line interface
8. **Notebooks** (`notebooks/explore.ipynb`) - Exploration notebook

### Platform Upgrade (New)
9. **Multi-Target Data Collector** (`src/data_collector.py`) - 8 protein targets (EGFR, BRAF, CDK2, HDAC1, HIV RT, ACE, COX-2, DRD2)
10. **Comprehensive Featurizer** (`src/featurizer.py`) - 6 methods + applicability domain
11. **Full Model Training Pipeline** (`src/trainer.py`) - 4 models, Optuna optimization, scaffold split, ensemble
12. **Model Explainability** (`src/explainer.py`) - SHAP global/local explanations
13. **Virtual Screening Engine** (`src/screener.py`) - Drug-likeness filtering, diversity, enrichment
14. **Chemical Space Visualizer** (`src/visualizer.py`) - UMAP, activity cliffs, scaffold analysis, plotly
15. **Chemical Similarity Search** (`src/similarity.py`) - Scaffold hopping, substructure search, MMP analysis
16. **Upgraded Predictor** (`src/predictor.py`) - Ensemble uncertainty, multitarget, selectivity, optimization
17. **Full CLI Rebuild** (`cli.py`) - Subcommands: predict, screen, search, train, explain, data
18. **7 New Notebooks** (`notebooks/01_data_exploration.ipynb` to `07_full_pipeline_demo.ipynb`)

## Installation

```bash
git clone https://github.com/mahimmazidul/dti.git
cd dti
pip install -r requirements.txt
```

**Note:** RDKit and PyTorch installation may vary by platform. See `requirements.txt` for details.

## Usage

### Legacy Commands (Backward Compatible)
```bash
python cli.py EGFR --mode target
python cli.py Imatinib --mode drug --safety
python cli.py EGFR --mode target --graph
python cli.py EGFR --mode target --repurpose --min-phase 2
```

### New Subcommands (Platform Upgrade)
```bash
# Predict affinity
python cli.py predict --smiles "CCO" --target egfr --explain --selectivity

# Virtual screening
python cli.py screen --library molecules.smi --target egfr --filter lipinski --top-n 100

# Chemical similarity search
python cli.py search --query "CCO" --library molecules.smi --threshold 0.7 --scaffold-hop

# Model training
python cli.py train --target egfr --all --method combined

# Explainability
python cli.py explain --smiles "CCO" --target egfr --global

# Data collection
python cli.py data --fetch all --limit 5000
```

## Platform Architecture

```
dti/
├── src/
│   ├── chembl_query.py          # Original ChEMBL module (do not modify)
│   ├── pubchem_query.py         # Original PubChem module (do not modify)
│   ├── analyzer.py              # Main logic (updated)
│   ├── openfda_query.py         # FDA safety module
│   ├── visualizer.py            # Chemical space visualizer (rewritten)
│   ├── repurposing.py           # Repurposing scanner
│   ├── data_collector.py        # Multi-target ChEMBL downloader
│   ├── featurizer.py            # 6 featurization methods + applicability domain
│   ├── trainer.py               # 4 models, Optuna, scaffold split, ensemble
│   ├── explainer.py             # SHAP global/local explanations
│   ├── screener.py              # Virtual screening engine
│   ├── similarity.py            # Scaffold hopping, substructure search, MMP
│   ├── predictor.py             # Ensemble uncertainty, multitarget, selectivity, optimization
│   └── cli.py                   # Full CLI rebuild
├── data/
│   ├── raw/                     # ChEMBL CSVs per target
│   ├── processed/               # Featurized numpy arrays
│   └── .gitignore               # Ignore all data files
├── models/
│   ├── egfr/
│   ├── braf/
│   ├── cdk2/
│   ├── hdac1/
│   ├── hiv_rt/
│   ├── ace/
│   ├── cox2/
│   ├── drd2/
│   └── .gitignore               # Ignore all model files
├── notebooks/
│   ├── explore.ipynb            # Original exploration notebook (updated)
│   ├── 01_data_exploration.ipynb
│   ├── 02_featurization.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_explainability.ipynb
│   ├── 05_virtual_screening.ipynb
│   ├── 06_chemical_space.ipynb
│   └── 07_full_pipeline_demo.ipynb
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Dependencies
├── README.md                    # Documentation (this file)
└── HISTORY.md                   # Change history
```

## Requirements

- Python 3.8+
- RDKit (platform-specific installation)
- PyTorch (platform-specific installation)
- scikit-learn, xgboost, lightgbm
- numpy, pandas, matplotlib, seaborn
- joblib, tqdm, optuna, shap, umap-learn
- scipy, statsmodels, deepchem, mordred, molvs
- chembl_webresource_client, pubchempy, requests, tabulate
- jupyter, ipywidgets, plotly, networkx

## API Rate Limits

- OpenFDA: 0.5s sleep between requests
- Repurposing: 0.3s sleep between requests
- CLI FDA section: 1s sleep

## License

MIT License - See LICENSE file for details

## Change History

See `HISTORY.md` for complete change log.

## Tags

- drug-discovery
- cheminformatics
- machine-learning
- chembl
- pubchem
- openfda
- virtual-screening
- similarity-search
- explainability
- optimization
- python

## Description

A command-line tool for exploring drug-target interactions using ChEMBL and PubChem.
Built with free public APIs and Python. Upgraded into a full professional-grade
cheminformatics and machine learning platform for drug discovery with 8 protein targets,
multi-target selectivity profiling, scaffold hopping, matched molecular pair analysis,
and full explainability.

## Repository Status

**Latest Commit:** 1973180a72a42281ecc433763b2a1bdb8c139312  
**Commit Message:** add: notebook 07 full pipeline demo end-to-end  
**Date:** 2026-11-08  

All features completed and pushed separately. One commit per step.
