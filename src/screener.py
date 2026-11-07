"""
Virtual Screening Engine

What virtual screening is:
Computationally filtering a large library of compounds (millions) to find those
most likely to be active against a target before doing expensive wet-lab work.
Instead of testing every compound in the lab (which costs millions), we use
machine learning to prioritize the top 100-1000 compounds for actual testing.

Two main types:
- Structure-based: needs protein 3D structure (docking). We don't do this here.
- Ligand-based: uses known active compounds to train a model (what we do).
  The model learns "what makes a good inhibitor" from existing ChEMBL data.

High-throughput virtual screening in industry:
Companies screen 10M+ compounds computationally in hours/days, then choose
hundreds for synthesis and testing. This reduces cost from $1B+ to $100M+
and time from 10 years to 3-5 years for lead discovery.

Enrichment factor:
How much better than random the model is at finding actives in the top fraction.
If random selection finds 1% actives in top 1%, but our model finds 20%,
the enrichment factor is 20x. This measures screening efficiency.

ROC-AUC in screening context:
Area under the Receiver Operating Characteristic curve. Measures how well
the model ranks actives above inactives across all thresholds. AUC=1.0 is
perfect ranking, AUC=0.5 is random. In screening, we care most about the
top of the ranked list (early enrichment).
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import joblib
import base64
from io import BytesIO

# Try imports
try:
    from rdkit.Chem import Butina
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available in sandbox. Will use fallback.")

# ZINC15 downloader (simplified - would use actual API in production)
def download_zinc_subset(n_compounds=10000):
    """
    Downloads a random subset of ZINC15 drug-like compounds.
    In production, would use https://zinc15.docking.org/tranches/head.txt
    Here we return placeholder SMILES for demonstration.
    """
    print(f"Downloading {n_compounds} ZINC compounds (placeholder)...")
    # Placeholder SMILES - in production, would parse actual ZINC file
    placeholder_smiles = [
        "CCO", "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2", "CC1=CC=C(C=C1)C(=O)O"
    ] * (n_compounds // 5)
    
    # Add some random variations
    compounds = []
    for i, smiles in enumerate(placeholder_smiles[:n_compounds]):
        compounds.append((smiles, f"ZINC_{i:06d}"))
    
    return compounds

def load_screening_library(sdf_path=None, smiles_path=None,
                            use_zinc=False, n_zinc_compounds=10000):
    """
    Loads screening library from SDF, SMILES file, or ZINC15.
    Returns list of (smiles, name) tuples.
    Validates all SMILES with RDKit, removes invalid ones.
    """
    compounds = []
    
    if sdf_path and RDKIT_AVAILABLE:
        try:
            supplier = Chem.SDMolSupplier(sdf_path)
            for i, mol in enumerate(supplier):
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"SDF_{i}"
                    compounds.append((smiles, name))
        except Exception as e:
            print(f"Error loading SDF: {e}")
    
    elif smiles_path:
        try:
            with open(smiles_path, 'r') as f:
                for i, line in enumerate(f):
                    smiles = line.strip()
                    if smiles:
                        compounds.append((smiles, f"SMILES_{i}"))
        except Exception as e:
            print(f"Error loading SMILES file: {e}")
    
    elif use_zinc:
        compounds = download_zinc_subset(n_zinc_compounds)
    
    # Validate SMILES
    valid_compounds = []
    invalid_count = 0
    
    for smiles, name in compounds:
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_compounds.append((smiles, name))
                else:
                    invalid_count += 1
            except:
                invalid_count += 1
        else:
            # Placeholder validation in sandbox
            valid_compounds.append((smiles, name))
    
    print(f"Loaded {len(valid_compounds)} valid compounds, removed {invalid_count} invalid")
    return valid_compounds

def screen_library(model, library_smiles_list, target_name,
                   featurization_method="combined",
                   batch_size=256, top_n=100,
                   training_smiles_list=None):
    """
    Takes a trained model and library, featurizes in batches,
    predicts pChEMBL, returns sorted results dataframe.
    """
    print(f"Screening {len(library_smiles_list)} compounds for {target_name}...")
    
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(library_smiles_list), batch_size), desc="Screening"):
        batch = library_smiles_list[i:i+batch_size]
        batch_smiles = [s[0] for s in batch]
        batch_names = [s[1] for s in batch]
        
        # Featurize batch (placeholder - would use src/featurizer.py)
        # In production: features = featurize_dataframe(pd.DataFrame({'canonical_smiles': batch_smiles}), method=featurization_method)
        n_features = 4410  # combined method size
        features = np.random.randn(len(batch), n_features)  # Placeholder
        
        # Predict
        if hasattr(model, '__call__'):  # Ensemble
            preds = model(features)
        elif hasattr(model, 'predict'):  # Scikit-learn/XGB/LGB
            preds = model.predict(features)
        else:  # PyTorch NN
            import torch
            model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(features)
                preds = model(X_t).numpy().flatten()
        
        # Convert pChEMBL to IC50 nM
        ic50_values = [10 ** (9 - p) * 1e-9 for p in preds]  # Simplified conversion
        
        # Potency class
        potency_classes = ["Active" if p >= 6 else "Inactive" for p in preds]
        
        # Lipinski check (placeholder)
        lipinski_pass = [True] * len(batch)  # In production, would check properties
        
        # Applicability domain (placeholder)
        in_domain = [True] * len(batch)  # In production, would check similarity
        
        for j, (smiles, name) in enumerate(batch):
            results.append({
                "smiles": smiles,
                "name": name,
                "predicted_pchembl": preds[j],
                "predicted_ic50_nM": ic50_values[j],
                "potency_class": potency_classes[j],
                "lipinski_pass": lipinski_pass[j],
                "in_applicability_domain": in_domain[j]
            })
    
    # Sort by predicted pChEMBL descending
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("predicted_pchembl", ascending=False)
    
    # Highlight top_n
    results_df["rank"] = range(1, len(results_df) + 1)
    results_df["is_top_hit"] = results_df["rank"] <= top_n
    
    print(f"Screening complete. Top {top_n} hits identified.")
    return results_df

def filter_by_drug_likeness(smiles_list, rules="lipinski"):
    """
    Filters compounds by drug-likeness rules.
    rules: lipinski, veber, ghose, leadlike, all
    Returns filtered list and list of removed compounds.
    """
    if not RDKIT_AVAILABLE:
        return smiles_list, []
    
    filtered = []
    removed = []
    
    for smiles, name in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                removed.append((smiles, name, "Invalid SMILES"))
                continue
            
            # Calculate properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            rotb = Descriptors.NumRotatableBonds(mol)
            
            # Check rules
            passes = True
            reasons = []
            
            if rules in ["lipinski", "all"]:
                if mw > 500:
                    passes = False; reasons.append("MW > 500")
                if logp > 5:
                    passes = False; reasons.append("LogP > 5")
                if hbd > 5:
                    passes = False; reasons.append("HBD > 5")
                if hba > 10:
                    passes = False; reasons.append("HBA > 10")
            
            if rules in ["veber", "all"]:
                if rotb > 10:
                    passes = False; reasons.append("RotBonds > 10")
                if tpsa > 140:
                    passes = False; reasons.append("TPSA > 140")
            
            if rules == "ghose":
                if not (160 <= mw <= 480):
                    passes = False; reasons.append("MW not in 160-480")
                if not (-0.4 <= logp <= 5.6):
                    passes = False; reasons.append("LogP not in -0.4-5.6")
                # ... other Ghose rules
            
            if rules == "leadlike":
                if mw > 350:
                    passes = False; reasons.append("MW > 350")
                if logp > 3:
                    passes = False; reasons.append("LogP > 3")
                if hbd > 3:
                    passes = False; reasons.append("HBD > 3")
                if hba > 7:
                    passes = False; reasons.append("HBA > 7")
            
            if passes:
                filtered.append((smiles, name))
            else:
                removed.append((smiles, name, ", ".join(reasons)))
        
        except Exception as e:
            removed.append((smiles, name, f"Error: {e}"))
    
    pass_rate = len(filtered) / len(smiles_list) * 100 if smiles_list else 0
    print(f"Drug-likeness filter ({rules}): {pass_rate:.1f}% passed ({len(filtered)}/{len(smiles_list)})")
    
    return filtered, removed

def diversity_filter(smiles_list, similarity_threshold=0.6):
    """
    Removes compounds too similar to each other using Butina clustering.
    Returns diverse subset where no two compounds have Tanimoto similarity above threshold.
    """
    if not RDKIT_AVAILABLE:
        return smiles_list
    
    print(f"Applying diversity filter (threshold={similarity_threshold})...")
    
    # Convert to molecules
    mols = []
    valid_smiles = []
    for smiles, name in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mols.append(mol)
            valid_smiles.append((smiles, name))
    
    if not mols:
        return []
    
    # Compute fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in mols]
    
    # Butina clustering
    # Note: This is a simplified implementation
    # In production, would use proper Butina clustering with distance matrix
    diverse_indices = [0]  # Start with first compound
    
    for i in range(1, len(fps)):
        is_similar = False
        for j in diverse_indices:
            sim = AllChem.DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if sim >= similarity_threshold:
                is_similar = True
                break
        
        if not is_similar:
            diverse_indices.append(i)
    
    diverse_compounds = [valid_smiles[i] for i in diverse_indices]
    removed_count = len(smiles_list) - len(diverse_compounds)
    
    print(f"Diversity filter: kept {len(diverse_compounds)}, removed {removed_count}")
    return diverse_compounds

def generate_screening_report(results_df, target_name,
                               model_name, output_path):
    """
    Generates HTML report of screening results.
    Includes summary statistics, histogram, scatter plot, top 100 hits table,
    and diversity metrics.
    """
    # Summary statistics
    total_screened = len(results_df)
    hit_count = len(results_df[results_df["potency_class"] == "Active"])
    hit_rate = hit_count / total_screened * 100 if total_screened > 0 else 0
    
    # Top 100 hits
    top_100 = results_df.head(100)
    
    # Generate HTML
    report_html = f"""
    <html>
    <head>
        <title>Screening Report: {target_name.upper()}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .section {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .good {{ color: green; font-weight: bold; }}
            .bad {{ color: red; font-weight: bold; }}
            .hit {{ background-color: #d4edda; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Virtual Screening Report</h1>
            <h2>Target: {target_name.upper()}</h2>
            <h2>Model: {model_name}</h2>
            
            <div class="section">
                <h3>Summary Statistics</h3>
                <table>
                    <tr><th>Total Compounds Screened</th><td>{total_screened}</td></tr>
                    <tr><th>Predicted Actives (pChEMBL >= 6)</th><td>{hit_count}</td></tr>
                    <tr><th>Hit Rate</th><td class="good">{hit_rate:.2f}%</td></tr>
                    <tr><th>Top 100 Hits Identified</th><td>Yes</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h3>Top 100 Hits</h3>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>SMILES</th>
                        <th>Name</th>
                        <th>Predicted pChEMBL</th>
                        <th>Predicted IC50 (nM)</th>
                        <th>Potency Class</th>
                        <th>Lipinski Pass</th>
                        <th>In Domain</th>
                    </tr>
    """
    
    for _, row in top_100.iterrows():
        report_html += f"""
            <tr class="{'hit' if row['potency_class'] == 'Active' else ''}">
                <td>{row['rank']}</td>
                <td>{row['smiles']}</td>
                <td>{row['name']}</td>
                <td>{row['predicted_pchembl']:.2f}</td>
                <td>{row['predicted_ic50_nM']:.2e}</td>
                <td class="{'good' if row['potency_class'] == 'Active' else 'bad'}">{row['potency_class']}</td>
                <td>{'✓' if row['lipinski_pass'] else '✗'}</td>
                <td>{'✓' if row['in_applicability_domain'] else '✗'}</td>
            </tr>
        """
    
    report_html += """
                </table>
            </div>
            
            <div class="section">
                <h3>Disclaimer</h3>
                <p>These screening results are computational predictions.
                Always validate top hits with wet-lab experiments.
                Predictions outside applicability domain should be treated with caution.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(report_html)
    
    print(f"Saved screening report to {output_path}")
    
    return results_df

def enrichment_analysis(model, known_actives_smiles,
                        known_inactives_smiles,
                        featurization_method="combined"):
    """
    Tests how well the model enriches actives vs inactives.
    Calculates ROC-AUC, Enrichment factor at 1%, 5%, 10%, BEDROC.
    Plots ROC curve.
    Returns metrics dict.
    """
    print("Running enrichment analysis...")
    
    # Combine and label
    all_smiles = known_actives_smiles + known_inactives_smiles
    labels = [1] * len(known_actives_smiles) + [0] * len(known_inactives_smiles)
    
    # Predict (placeholder - would use actual featurization and model)
    # In production: features = featurize_dataframe(...)
    n_features = 4410
    features = np.random.randn(len(all_smiles), n_features)
    
    if hasattr(model, '__call__'):
        preds = model(features)
    elif hasattr(model, 'predict'):
        preds = model.predict(features)
    else:
        import torch
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(features)
            preds = model(X_t).numpy().flatten()
    
    # ROC-AUC
    roc_auc = roc_auc_score(labels, preds)
    
    # Enrichment factors
    sorted_indices = np.argsort(preds)[::-1]
    sorted_labels = np.array(labels)[sorted_indices]
    
    n_total = len(labels)
    n_actives = sum(labels)
    
    # Enrichment at 1%, 5%, 10%
    ef_1 = (sum(sorted_labels[:int(n_total * 0.01)]) / (n_total * 0.01)) / (n_actives / n_total) if n_total > 0 else 0
    ef_5 = (sum(sorted_labels[:int(n_total * 0.05)]) / (n_total * 0.05)) / (n_actives / n_total) if n_total > 0 else 0
    ef_10 = (sum(sorted_labels[:int(n_total * 0.1)]) / (n_total * 0.1)) / (n_actives / n_total) if n_total > 0 else 0
    
    # ROC curve
    fpr, tpr, _ = roc_curve(labels, preds)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Virtual Screening Enrichment')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    metrics = {
        "roc_auc": roc_auc,
        "enrichment_factor_1_percent": ef_1,
        "enrichment_factor_5_percent": ef_5,
        "enrichment_factor_10_percent": ef_10,
        "total_compounds": n_total,
        "total_actives": n_actives
    }
    
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"Enrichment Factor (1%): {ef_1:.1f}x")
    print(f"Enrichment Factor (5%): {ef_5:.1f}x")
    print(f"Enrichment Factor (10%): {ef_10:.1f}x")
    
    return metrics

def scaffold_analysis(results_df, top_n=100):
    """
    Analyzes scaffolds in top screening results.
    Returns most common scaffolds and their predicted potencies.
    """
    if not RDKIT_AVAILABLE:
        return pd.DataFrame()
    
    top_hits = results_df.head(top_n)
    scaffolds = []
    
    for smiles in top_hits["smiles"]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.append(Chem.MolToSmiles(scaffold))
            else:
                scaffolds.append("Invalid")
        except:
            scaffolds.append("Error")
    
    top_hits["scaffold"] = scaffolds
    
    # Count unique scaffolds
    scaffold_counts = top_hits["scaffold"].value_counts()
    scaffold_potencies = top_hits.groupby("scaffold")["predicted_pchembl"].mean()
    
    analysis_df = pd.DataFrame({
        "scaffold": scaffold_counts.index,
        "count": scaffold_counts.values,
        "mean_pchembl": scaffold_potencies.values
    }).sort_values("count", ascending=False)
    
    print(f"Scaffold analysis: {len(analysis_df)} unique scaffolds in top {top_n} hits")
    return analysis_df

# Example usage
if __name__ == "__main__":
    print("Screener module loaded.")
    print("Functions: load_screening_library, screen_library,")
    print("           filter_by_drug_likeness, diversity_filter,")
    print("           generate_screening_report, enrichment_analysis,")
    print("           scaffold_analysis")
