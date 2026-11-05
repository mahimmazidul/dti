# Drug-Target Interaction (DTI) Analyzer

Drug-target interaction describes how a drug molecule binds to and affects a specific protein or biological target in the body. Understanding these interactions helps researchers predict which drugs might treat which diseases and how potent those drugs will be.

## What this tool does

The DTI Analyzer is a command-line tool built with Python that lets you explore drug-target interactions using free public APIs. You can search for targets (like proteins) and find drugs that interact with them, or search for drugs and see their properties and targets.

The tool now has four capabilities:
- Search drug and target interactions by compound or protein name
- Retrieve FDA drug safety and adverse event data
- Visualize interaction networks as weighted graphs
- Scan for drug repurposing candidates by clinical phase

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mahimmazidul/dti.git
   cd dti
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## CLI Usage Examples

### Search for drugs targeting a protein (default mode)
```bash
python cli.py EGFR --mode target --limit 10
```
**Sample Output:**
```
DTI Analyzer - mahimmazidul
Data sources: ChEMBL, PubChem (free, open access)

+------------------+----------+----------+---------------------------------+
| Drug Name        | IC50 (nM)| pChEMBL  | SMILES                          |
+------------------+----------+----------+---------------------------------+
| CHEMBL12345      | 12.5     | 7.9      | Cc1ccc(cc1)NC(=O)c2ccc(cc2)F    |
| CHEMBL67890      | 45.0     | 7.3      | Cc1ccc(cc1)NC(=O)c2ccc(cc2)Cl   |
| ...              | ...      | ...      | ...                             |
+------------------+----------+----------+---------------------------------+
```

### Search for properties of a drug
```bash
python cli.py Imatinib --mode drug
```
**Sample Output:**
```
DTI Analyzer - mahimmazidul
Data sources: ChEMBL, PubChem (free, open access)

+------------------+------------------+--------+---------------------------------+------------------+
| Molecular Weight | Formula          | XLogP  | SMILES                          | Max Phase        |
+------------------+------------------+--------+---------------------------------+------------------+
| 493.60           | C29H31N7O        | 4.5    | Cc1ccc(cc1)NC(=O)c2ccc(cc2)F    | 4                |
+------------------+------------------+--------+---------------------------------+------------------+
```

### Save results to CSV
```bash
python cli.py BRCA1 --mode target --save results.csv
```

### Network Graph Example
The network graph visualizes drugs and proteins as connected points, showing interaction strength through line thickness. This helps identify polypharmacology patterns and potential drug repurposing opportunities at a glance.
```bash
python cli.py EGFR --mode target --graph --save-graph egfr_network.png
```

## Data Sources

This tool uses only free and open public APIs:
- **ChEMBL**: A database of bioactive molecules with drug-like properties, maintained by EMBL-EBI.
- **PubChem**: A database of chemical molecules and their activities against biological assays, maintained by NCBI.

## Roadmap

- OpenFDA safety data (done, use --safety flag with drug mode)
- Network graph visualization (done, use --graph flag with target mode)
- Drug repurposing scanner (done, use --repurpose flag with target mode)
- Web interface using Flask or Streamlit

## Drug Repurposing Scanner
This feature scans for approved or clinical-stage drugs that interact with a target protein, helping identify new uses for existing medications. It is useful because repurposing existing drugs can save years of development time and billions in costs compared to creating new drugs from scratch.
```bash
python cli.py EGFR --mode target --repurpose --min-phase 2
```

---
Created by Mazidul Islam Mahim
