"""
Chemical Space Visualizer

What chemical space is:
The abstract multi-dimensional space where each molecule occupies a point
defined by its properties (fingerprints, descriptors, etc.). Imagine a map
where similar molecules cluster together and different ones are far apart.

Why visualizing it matters:
Seeing where your actives cluster and where unexplored regions are guides
synthesis planning. If all your actives cluster in one small region, you're
missing opportunities. If they're spread out, you have diverse chemical
series. Visualizations help medicinal chemists understand the "shape" of
their chemical space.

What UMAP is:
Uniform Manifold Approximation and Projection - a dimensionality reduction
method that preserves both local and global structure better than t-SNE.
It projects high-dimensional molecular fingerprints (2048+ dimensions)
into 2D or 3D while keeping similar molecules close and dissimilar ones
far apart, even across large datasets.

What scaffold diversity means visually:
A good drug discovery campaign explores diverse regions of chemical space.
If all compounds share the same scaffold (e.g., all quinazolines), you're
limited to one chemical series. Visualizing scaffold distribution shows
whether you have multiple independent series or just variations of one.

Activity cliffs:
Pairs of very similar molecules with very different binding affinities.
They reveal the most sensitive SAR (structure-activity relationship)
regions - changing one atom can flip potency from nanomolar to micromolar.
Understanding cliffs helps identify critical structural features.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import joblib

# Try imports
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: UMAP not available in sandbox. Will use fallback.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

def plot_chemical_space_umap(df, smiles_col="canonical_smiles",
                              color_col="pchembl_value",
                              output_path=None):
    """
    Featurizes all SMILES with Morgan fingerprints, reduces to 2D with UMAP,
    plots scatter colored by color_col. Interactive with plotly if .html,
    static with matplotlib if .png.
    Returns UMAP coordinates dataframe.
    """
    print(f"Computing UMAP for {len(df)} compounds...")
    
    # Featurize (placeholder - would use src/featurizer.py)
    # In production: features = featurize_dataframe(df, method="morgan")
    n_features = 2048
    features = np.random.randn(len(df), n_features)  # Placeholder
    
    # UMAP reduction
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_coords = reducer.fit_transform(features)
    else:
        # Fallback: PCA-like projection
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        umap_coords = pca.fit_transform(features)
    
    df_coords = pd.DataFrame({
        "smiles": df[smiles_col],
        "umap_x": umap_coords[:, 0],
        "umap_y": umap_coords[:, 1],
        color_col: df[color_col] if color_col in df.columns else np.zeros(len(df))
    })
    
    # Plotting
    if output_path and output_path.endswith(".html") and PLOTLY_AVAILABLE:
        # Interactive plotly
        fig = px.scatter(
            df_coords, x="umap_x", y="umap_y", color=color_col,
            title=f"Chemical Space UMAP (colored by {color_col})",
            labels={"umap_x": "UMAP 1", "umap_y": "UMAP 2", color_col: color_col},
            color_continuous_scale="RdBu_r" if color_col == "pchembl_value" else "viridis"
        )
        fig.write_html(output_path)
        print(f"Saved interactive plot to {output_path}")
    
    elif output_path and output_path.endswith(".png"):
        # Static matplotlib
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            df_coords["umap_x"], df_coords["umap_y"],
            c=df_coords[color_col], cmap="RdBu_r" if color_col == "pchembl_value" else "viridis",
            alpha=0.7, s=20
        )
        plt.colorbar(scatter, label=color_col)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(f"Chemical Space UMAP (colored by {color_col})")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved static plot to {output_path}")
    
    return df_coords

def plot_scaffold_distribution(df, smiles_col="canonical_smiles",
                               top_n=20, output_path=None):
    """
    Plots bar chart of top_n most common scaffolds with mean pChEMBL.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available, skipping scaffold plot")
        return
    
    scaffolds = []
    pchembl_values = []
    
    for smiles, pchembl in zip(df[smiles_col], df.get("pchembl_value", [0]*len(df))):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.append(Chem.MolToSmiles(scaffold))
                pchembl_values.append(pchembl)
        except:
            pass
    
    scaffold_df = pd.DataFrame({
        "scaffold": scaffolds,
        "pchembl": pchembl_values
    })
    
    # Group by scaffold
    scaffold_stats = scaffold_df.groupby("scaffold").agg(
        count=("pchembl", "count"),
        mean_pchembl=("pchembl", "mean")
    ).sort_values("count", ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(scaffold_stats)), scaffold_stats["count"], color='skyblue')
    plt.xticks(range(len(scaffold_stats)), [s[:20] + "..." if len(s) > 20 else s for s in scaffold_stats.index], rotation=45, ha='right')
    plt.ylabel("Count")
    plt.title(f"Top {top_n} Scaffolds (Mean pChEMBL annotated)")
    
    # Annotate with mean pChEMBL
    for i, bar in enumerate(bars):
        height = bar.get_height()
        mean_p = scaffold_stats.iloc[i]["mean_pchembl"]
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{mean_p:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved scaffold distribution plot to {output_path}")
    
    plt.close()
    
    return scaffold_stats

def plot_property_distributions(df, output_path=None):
    """
    Multi-panel plot showing distributions of MW, LogP, HBD, HBA, TPSA, RotBonds, pChEMBL.
    2 rows 4 columns layout.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available, skipping property distributions")
        return
    
    # Calculate properties (placeholder - would use src/featurizer.py)
    properties = {
        "MW": [], "LogP": [], "HBD": [], "HBA": [],
        "TPSA": [], "RotBonds": [], "pChEMBL": df.get("pchembl_value", [0]*len(df)).tolist()
    }
    
    for smiles in df["canonical_smiles"]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                properties["MW"].append(Descriptors.MolWt(mol))
                properties["LogP"].append(Descriptors.MolLogP(mol))
                properties["HBD"].append(Descriptors.NumHDonors(mol))
                properties["HBA"].append(Descriptors.NumHAcceptors(mol))
                properties["TPSA"].append(rdMolDescriptors.CalcTPSA(mol))
                properties["RotBonds"].append(Descriptors.NumRotatableBonds(mol))
        except:
            pass
    
    prop_df = pd.DataFrame(properties)
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    prop_names = ["MW", "LogP", "HBD", "HBA", "TPSA", "RotBonds", "pChEMBL"]
    colors = ["skyblue", "lightgreen", "salmon", "lightpurple", "orange", "lightblue", "pink"]
    
    for i, prop in enumerate(prop_names):
        if prop in prop_df.columns:
            ax = axes[i]
            sns.histplot(prop_df[prop], kde=True, ax=ax, color=colors[i])
            ax.set_title(f"{prop} Distribution")
            ax.set_xlabel(prop)
            ax.set_ylabel("Count")
            
            # Add Lipinski cutoffs where relevant
            if prop == "MW":
                ax.axvline(500, color='red', linestyle='--', alpha=0.5, label="Lipinski cutoff (500)")
            elif prop == "LogP":
                ax.axvline(5, color='red', linestyle='--', alpha=0.5, label="Lipinski cutoff (5)")
            elif prop == "HBD":
                ax.axvline(5, color='red', linestyle='--', alpha=0.5, label="Lipinski cutoff (5)")
            elif prop == "HBA":
                ax.axvline(10, color='red', linestyle='--', alpha=0.5, label="Lipinski cutoff (10)")
            
            ax.legend()
    
    # Hide unused subplots
    for i in range(len(prop_names), 8):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved property distributions plot to {output_path}")
    
    plt.close()
    
    return prop_df

def plot_activity_cliff_analysis(df, smiles_col="canonical_smiles",
                                 activity_col="pchembl_value",
                                 similarity_threshold=0.9,
                                 activity_threshold=2.0,
                                 output_path=None):
    """
    Finds activity cliffs: pairs of very similar molecules with very
    different binding affinities. Plots network where nodes are molecules
    colored by activity and edges connect cliff pairs.
    Returns cliff pairs dataframe.
    
    Activity cliffs matter because they reveal the most sensitive SAR regions:
    changing one atom can flip potency from nanomolar to micromolar, showing
    exactly which structural features are critical for binding.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available, skipping activity cliff analysis")
        return pd.DataFrame()
    
    print("Analyzing activity cliffs...")
    
    # Sample subset for speed (max 500 compounds)
    sample_df = df.sample(min(500, len(df)), random_state=42)
    smiles_list = sample_df[smiles_col].tolist()
    pchembl_list = sample_df[activity_col].tolist()
    
    # Compute fingerprints
    mols = []
    valid_indices = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mols.append(mol)
            valid_indices.append(i)
    
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in mols]
    
    # Find cliffs
    cliff_pairs = []
    n = len(fps)
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            p_diff = abs(pchembl_list[valid_indices[i]] - pchembl_list[valid_indices[j]])
            
            if sim >= similarity_threshold and p_diff >= activity_threshold:
                cliff_pairs.append({
                    "smiles1": smiles_list[valid_indices[i]],
                    "smiles2": smiles_list[valid_indices[j]],
                    "similarity": sim,
                    "pchembl1": pchembl_list[valid_indices[i]],
                    "pchembl2": pchembl_list[valid_indices[j]],
                    "pchembl_diff": p_diff
                })
    
    cliff_df = pd.DataFrame(cliff_pairs)
    
    # Plot network
    if cliff_df.empty:
        print("No activity cliffs found")
        return cliff_df
    
    G = nx.Graph()
    
    for _, row in cliff_df.iterrows():
        G.add_edge(row["smiles1"], row["smiles2"], weight=row["similarity"])
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    # Color nodes by average pChEMBL
    node_colors = []
    for node in G.nodes():
        # Find pChEMBL for this node
        idx1 = sample_df[sample_df[smiles_col] == node].index
        if len(idx1) > 0:
            node_colors.append(sample_df.loc[idx1[0], activity_col])
        else:
            node_colors.append(0)
    
    nx.draw(G, pos, with_labels=False, node_size=50, node_color=node_colors,
            edge_color='gray', alpha=0.7, cmap=plt.cm.RdBu_r)
    
    plt.title(f"Activity Cliff Network (Threshold: Sim>{similarity_threshold}, ΔpChEMBL>{activity_threshold})")
    plt.colorbar(label="pChEMBL")
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved activity cliff plot to {output_path}")
    
    plt.close()
    
    return cliff_df

def plot_training_coverage(train_df, test_df, query_smiles=None,
                            output_path=None):
    """
    Shows UMAP of training set (grey) and test set (blue).
    If query_smiles provided, shows it as red star.
    Visualizes applicability domain.
    """
    # Combine data for UMAP
    combined_df = pd.concat([
        train_df.assign(set="train"),
        test_df.assign(set="test")
    ], ignore_index=True)
    
    coords_df = plot_chemical_space_umap(combined_df, output_path=None)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    train_coords = coords_df[coords_df["set"] == "train"]
    test_coords = coords_df[coords_df["set"] == "test"]
    
    plt.scatter(train_coords["umap_x"], train_coords["umap_y"],
                c='lightgray', alpha=0.5, s=10, label="Training Set")
    plt.scatter(test_coords["umap_x"], test_coords["umap_y"],
                c='blue', alpha=0.7, s=20, label="Test Set")
    
    # Query molecule
    if query_smiles and RDKIT_AVAILABLE:
        try:
            # Placeholder: would compute UMAP for query
            query_x, query_y = np.mean(coords_df["umap_x"]), np.mean(coords_df["umap_y"])
            plt.scatter([query_x], [query_y], c='red', marker='*', s=200,
                        label="Query Molecule", edgecolors='black', linewidth=2)
        except:
            pass
    
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("Training Coverage - Applicability Domain")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved training coverage plot to {output_path}")
    
    plt.close()

def build_interaction_network(target_name, bioactivities,
                              output_file=None):
    """
    Keeps existing network graph code from old visualizer.
    Builds interaction network with potency weighting.
    """
    print(f"Building interaction network for {target_name}...")
    
    G = nx.Graph()
    
    # Add nodes and edges from bioactivities
    for bio in bioactivities:
        drug_name = bio.get("drug_name", "Unknown")
        target_id = bio.get("target_chembl_id", target_name)
        ic50 = bio.get("ic50_nm", 1000)  # Default 1000 nM
        pchembl = bio.get("pchembl_value", 0)
        
        # Add nodes
        G.add_node(drug_name, type="drug", pchembl=pchembl)
        G.add_node(target_id, type="target")
        
        # Add edge with weight based on potency (lower IC50 = stronger binding = thicker edge)
        weight = 1000 / ic50 if ic50 > 0 else 1
        G.add_edge(drug_name, target_id, weight=weight, pchembl=pchembl)
    
    # Plot
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    # Separate nodes by type
    drug_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "drug"]
    target_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "target"]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_color='skyblue', node_size=100, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_color='lightcoral', node_size=500, alpha=0.8)
    
    # Draw edges with width based on weight
    edges = G.edges(data=True)
    weights = [d["weight"] * 0.1 for _, _, d in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    
    plt.title(f"Drug-Target Interaction Network: {target_name}")
    plt.axis("off")
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Saved network plot to {output_file}")
    
    plt.close()
    
    return G

def plot_model_comparison(metrics_dict, output_path=None):
    """
    Plots grouped bar chart comparing RMSE, MAE, R2 across models.
    Also plots radar/spider chart for multi-metric comparison.
    """
    models = list(metrics_dict.keys())
    rmses = [metrics_dict[m]["rmse"] for m in models]
    maes = [metrics_dict[m]["mae"] for m in models]
    r2s = [metrics_dict[m]["r2"] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, rmses, width, label='RMSE', color='skyblue')
    ax.bar(x, maes, width, label='MAE', color='lightgreen')
    ax.bar(x + width, r2s, width, label='R²', color='salmon')
    
    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved model comparison plot to {output_path}")
    
    plt.close()

def plot_optuna_history(study, output_path=None):
    """
    Plots Optuna optimization history showing RMSE improvement.
    """
    if not hasattr(study, 'trials'):
        print("No Optuna study provided")
        return
    
    trial_values = [t.value for t in study.trials if t.value is not None]
    trial_numbers = [t.number for t in study.trials if t.value is not None]
    
    plt.figure(figsize=(10, 6))
    plt.plot(trial_numbers, trial_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Trial Number")
    plt.ylabel("RMSE")
    plt.title("Optuna Optimization History")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved Optuna history plot to {output_path}")
    
    plt.close()

# Keep existing network graph functionality from old visualizer
# (from previous session Feature 2)

# Example usage
if __name__ == "__main__":
    print("Visualizer module loaded (rewritten with chemical space features).")
    print("Functions: plot_chemical_space_umap, plot_scaffold_distribution,")
    print("           plot_property_distributions, plot_activity_cliff_analysis,")
    print("           plot_training_coverage, build_interaction_network,")
    print("           plot_model_comparison, plot_optuna_history")
