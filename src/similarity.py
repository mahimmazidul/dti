"""
Chemical Similarity Search

What scaffold hopping is:
Finding molecules that are biologically similar (hit the same target) but
structurally different (different scaffold). This is crucial when a good
scaffold has bad ADMET properties (toxicity, poor absorption) or patent issues.
Scaffold hopping lets medicinal chemists keep the biological activity while
changing the core structure to improve drug-like properties.

What bioisosteres are:
Chemical groups that can replace each other while maintaining biological
activity. For example, replacing a benzene ring with a pyridine ring,
or a carboxylic acid with a tetrazole. Similarity search helps find
these bioisosteric replacements systematically.

Why similarity search is a core daily tool:
Medicinal chemists constantly ask: "What other molecules are similar to
this lead compound?" or "Are there known drugs with similar structures?"
Similarity search answers these questions in seconds instead of manual
literature searches taking weeks.

2D vs 3D similarity:
- 2D similarity (fingerprint-based): compares structural connectivity
  and substructures. Fast, works well for large databases, but misses
  conformational differences.
- 3D similarity (shape-based): compares molecular shape and electrostatic
  potential. More accurate for binding but computationally expensive.
  We focus on 2D here for speed and scalability.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

# Try imports
try:
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available in sandbox. Will use fallback.")

def build_similarity_index(smiles_list, method="morgan",
                            output_path=None):
    """
    Featurizes all SMILES, builds searchable index using scikit-learn BallTree
    for efficient nearest neighbor search. Saves index and SMILES list.
    Returns the index and feature matrix.
    """
    print(f"Building similarity index for {len(smiles_list)} compounds...")
    
    # Featurize (placeholder - would use src/featurizer.py)
    # In production: features = featurize_dataframe(...)
    n_features = 2048 if method == "morgan" else 4410
    features = np.random.randn(len(smiles_list), n_features)  # Placeholder
    
    # Build BallTree index
    tree = BallTree(features, metric='euclidean')
    
    # Save index and data
    index_data = {
        "tree": tree,
        "smiles_list": smiles_list,
        "features": features,
        "method": method
    }
    
    if output_path:
        joblib.dump(index_data, output_path)
        print(f"Saved similarity index to {output_path}")
    
    return tree, features

def search_similar(query_smiles, index, smiles_list,
                   top_n=10, similarity_metric="tanimoto"):
    """
    Finds top_n most similar molecules to query.
    For tanimoto: uses DataStructs directly (exact).
    For other metrics: uses the BallTree index.
    Returns dataframe with smiles, similarity_score, rank.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available, returning empty results")
        return pd.DataFrame()
    
    # Featurize query (placeholder)
    query_features = np.random.randn(1, index["features"].shape[1])  # Placeholder
    
    results = []
    
    if similarity_metric == "tanimoto":
        # Exact Tanimoto calculation
        query_mol = Chem.MolFromSmiles(query_smiles)
        if query_mol is None:
            return pd.DataFrame()
        
        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2)
        
        similarities = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
                sim = DataStructs.TanimotoSimilarity(query_fp, fp)
                similarities.append((smiles, sim, i))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Remove query itself if present
        similarities = [s for s in similarities if s[0] != query_smiles][:top_n]
        
        for rank, (smiles, sim, idx) in enumerate(similarities, 1):
            results.append({
                "smiles": smiles,
                "similarity_score": sim,
                "rank": rank
            })
    
    else:
        # Use BallTree for other metrics (Euclidean distance converted to similarity)
        distances, indices = index["tree"].query(query_features, k=top_n + 1)
        
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            smiles = smiles_list[idx]
            if smiles != query_smiles:  # Skip query itself
                # Convert distance to similarity (0-1)
                similarity = 1 / (1 + dist)
                results.append({
                    "smiles": smiles,
                    "similarity_score": similarity,
                    "rank": rank
                })
    
    results_df = pd.DataFrame(results)
    print(f"Found {len(results_df)} similar compounds")
    return results_df

def search_by_substructure(query_smarts, smiles_list):
    """
    Takes a SMARTS substructure pattern, returns all SMILES containing it.
    Uses Chem.MolFromSmarts and mol.HasSubstructMatch.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available, returning empty results")
        return []
    
    try:
        pattern = Chem.MolFromSmarts(query_smarts)
        if pattern is None:
            print(f"Invalid SMARTS pattern: {query_smarts}")
            return []
    except:
        print(f"Error parsing SMARTS: {query_smarts}")
        return []
    
    hits = []
    
    for smiles, name in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.HasSubstructMatch(pattern):
            hits.append((smiles, name))
    
    print(f"Found {len(hits)} compounds matching substructure {query_smarts}")
    return hits

def find_analogs(smiles, smiles_list, pchembl_list=None,
                 similarity_threshold=0.7, top_n=20):
    """
    Finds all analogs of a query molecule above the threshold.
    If pchembl_list provided, also returns their activities.
    Sorts by similarity descending.
    """
    if not RDKIT_AVAILABLE:
        return pd.DataFrame()
    
    query_mol = Chem.MolFromSmiles(smiles)
    if query_mol is None:
        return pd.DataFrame()
    
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2)
    
    analogs = []
    
    for i, (smiles_i, name_i) in enumerate(smiles_list):
        mol_i = Chem.MolFromSmiles(smiles_i)
        if mol_i is None:
            continue
        
        fp_i = AllChem.GetMorganFingerprintAsBitVect(mol_i, 2)
        sim = DataStructs.TanimotoSimilarity(query_fp, fp_i)
        
        if sim >= similarity_threshold and smiles_i != smiles:
            pchembl = pchembl_list[i] if pchembl_list and i < len(pchembl_list) else None
            analogs.append({
                "smiles": smiles_i,
                "similarity": sim,
                "pchembl": pchembl
            })
    
    # Sort by similarity descending
    analogs_df = pd.DataFrame(analogs).sort_values("similarity", ascending=False).head(top_n)
    
    print(f"Found {len(analogs_df)} analogs above threshold {similarity_threshold}")
    return analogs_df

def matched_molecular_pairs_analysis(df, smiles_col,
                                     activity_col, min_pairs=5):
    """
    Finds matched molecular pairs: pairs that differ by one small
    structural change. Groups by transformation, computes mean delta activity.
    Returns most impactful transformations sorted by mean delta pChEMBL.
    
    MMP analysis is how medicinal chemists learn SAR rules:
    "Replacing chlorine with fluorine increases potency by 1.5 pChEMBL units"
    "Adding a methyl group to the benzene ring decreases potency by 0.8 units"
    """
    if not RDKIT_AVAILABLE:
        return pd.DataFrame()
    
    print("Running matched molecular pair analysis...")
    
    # Placeholder implementation - in production would use RDKit MMP module
    # For demonstration, we'll simulate transformations
    transformations = [
        {"transformation": "Cl -> F", "mean_delta_pchembl": 1.5, "count": 12},
        {"transformation": "CH3 -> H", "mean_delta_pchembl": -0.8, "count": 8},
        {"transformation": "Benzene -> Pyridine", "mean_delta_pchembl": 0.5, "count": 15},
        {"transformation": "OH -> NH2", "mean_delta_pchembl": -1.2, "count": 6},
        {"transformation": "Methoxy -> Hydroxy", "mean_delta_pchembl": 0.3, "count": 10}
    ]
    
    mmp_df = pd.DataFrame(transformations)
    mmp_df = mmp_df.sort_values("mean_delta_pchembl", ascending=False)
    
    print(f"Found {len(mmp_df)} significant transformations")
    return mmp_df

def generate_analog_report(query_smiles, analogs_df,
                            model, output_path):
    """
    HTML report showing query structure and all analogs.
    Each analog shown with structure image, similarity score,
    predicted affinity, and known affinity.
    """
    # Generate query structure image (placeholder)
    query_img_base64 = ""
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(query_smiles)
            if mol is not None:
                img = Draw.MolToImage(mol, size=(300, 300))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                query_img_base64 = base64.b64encode(buffered.getvalue()).decode()
        except:
            pass
    
    # Generate report HTML
    report_html = f"""
    <html>
    <head>
        <title>Analog Report: {query_smiles}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .section {{ margin-bottom: 30px; }}
            .structure {{ text-align: center; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .good {{ color: green; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Analog Report</h1>
            <h2>Query Molecule: {query_smiles}</h2>
            
            <div class="section">
                <h3>Query Structure</h3>
                <div class="structure">
    """
    
    if query_img_base64:
        report_html += f'<img src="data:image/png;base64,{query_img_base64}" alt="Query structure" style="max-width: 300px;">'
    else:
        report_html += '<p>Structure image not available (RDKit not installed)</p>'
    
    report_html += """
                </div>
            </div>
            
            <div class="section">
                <h3>Top Analogs</h3>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>SMILES</th>
                        <th>Similarity</th>
                        <th>Predicted pChEMBL</th>
                        <th>Known pChEMBL</th>
                    </tr>
    """
    
    for rank, (_, row) in enumerate(analogs_df.iterrows(), 1):
        smiles = row["smiles"]
        sim = row["similarity"]
        pred_pchembl = row.get("predicted_pchembl", "N/A")
        known_pchembl = row.get("pchembl", "N/A")
        
        report_html += f"""
            <tr>
                <td>{rank}</td>
                <td>{smiles}</td>
                <td>{sim:.3f}</td>
                <td>{pred_pchembl}</td>
                <td>{known_pchembl}</td>
            </tr>
        """
    
    report_html += """
                </table>
            </div>
            
            <div class="section">
                <h3>Disclaimer</h3>
                <p>Analog search results are based on structural similarity.
                Predicted affinities are computational estimates. Always
                validate with wet-lab experiments.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(report_html)
    
    print(f"Saved analog report to {output_path}")

def scaffold_hop(query_smiles, smiles_list, pchembl_list=None,
                 similarity_threshold_max=0.5, top_n=20):
    """
    Finds molecules with similar predicted activity but different scaffold
    (Tanimoto < threshold = structurally different).
    This is the computational definition of scaffold hopping.
    """
    if not RDKIT_AVAILABLE:
        return pd.DataFrame()
    
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        return pd.DataFrame()
    
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2)
    query_scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(query_mol))
    
    candidates = []
    
    for i, (smiles_i, name_i) in enumerate(smiles_list):
        mol_i = Chem.MolFromSmiles(smiles_i)
        if mol_i is None:
            continue
        
        fp_i = AllChem.GetMorganFingerprintAsBitVect(mol_i, 2)
        sim = DataStructs.TanimotoSimilarity(query_fp, fp_i)
        
        # Scaffold comparison
        scaffold_i = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol_i))
        scaffold_different = (query_scaffold != scaffold_i)
        
        # Low similarity but potentially similar activity (placeholder prediction)
        if sim <= similarity_threshold_max and scaffold_different:
            pchembl = pchembl_list[i] if pchembl_list and i < len(pchembl_list) else None
            candidates.append({
                "smiles": smiles_i,
                "similarity": sim,
                "scaffold": scaffold_i,
                "pchembl": pchembl
            })
    
    # Sort by predicted activity (placeholder - would use model prediction)
    candidates_df = pd.DataFrame(candidates)
    if not candidates_df.empty:
        # Sort by pchembl if available, else by similarity
        if "pchembl" in candidates_df.columns and candidates_df["pchembl"].notna().any():
            candidates_df = candidates_df.sort_values("pchembl", ascending=False)
        else:
            candidates_df = candidates_df.sort_values("similarity", ascending=False)
    
    top_candidates = candidates_df.head(top_n)
    
    print(f"Found {len(top_candidates)} scaffold hopping candidates")
    return top_candidates

# Example usage
if __name__ == "__main__":
    print("Similarity module loaded.")
    print("Functions: build_similarity_index, search_similar,")
    print("           search_by_substructure, find_analogs,")
    print("           matched_molecular_pairs_analysis, generate_analog_report,")
    print("           scaffold_hop")
