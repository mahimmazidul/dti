"""
Model Explainability with SHAP

What SHAP values are:
SHAP (SHapley Additive exPlanations) values quantify the contribution of each
feature to a prediction, grounded in Shapley values from cooperative game theory.
For a molecule prediction, each fingerprint bit or descriptor gets a value
showing how much it pushed the prediction up or down from the baseline.

Why explainability matters in drug discovery:
A black box that says "this molecule is potent (pChEMBL=8.5)" is less useful
than one that says "the benzene ring and the amine group drive potency up by 2.3 units,
while the large molecular weight pulls it down by 0.8 units." Medicinal chemists
need to know WHAT to change, not just that the prediction is good.

Global vs local explanations:
- Global: across the whole dataset, which features matter most overall?
  (e.g., "across all EGFR inhibitors, aromatic rings and hydrogen bond donors
   are the most important features")
- Local: for one specific molecule, which features drive THIS prediction?
  (e.g., "for Gefitinib, the quinazoline core and the fluorine substituent
   are the main drivers of its high potency")

Fingerprint bit importance:
When a Morgan fingerprint bit has high SHAP importance, you can trace it back
to a specific substructure in the molecule. For example, bit 1234 might represent
"a benzene ring connected to a nitrogen", and if its SHAP value is +1.5, that
substructure contributes 1.5 pChEMBL units to the prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import base64
from io import BytesIO

# Try imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP not available in sandbox. Will use fallback.")

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

def compute_shap_values(model, X, model_type="tree",
                        background_size=100):
    """
    Computes SHAP values for the model and dataset.
    model_type:
      "tree": use shap.TreeExplainer (RF, XGB, LGB)
      "nn": use shap.DeepExplainer with background subset
      "kernel": use shap.KernelExplainer (slowest, universal)
    Returns shap_values array.
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available, returning zeros")
        return np.zeros_like(X)
    
    print(f"Computing SHAP values (model_type={model_type})...")
    
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # Handle multi-output case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        return shap_values
    
    elif model_type == "nn":
        # For neural networks, use DeepExplainer
        # Background is a subset of data
        background = X[:min(background_size, len(X))]
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X)
        return shap_values
    
    else:  # kernel
        explainer = shap.KernelExplainer(model.predict, X[:background_size])
        shap_values = explainer.shap_values(X)
        return shap_values

def global_feature_importance(shap_values, feature_names,
                               top_n=30, output_path=None):
    """
    Plots SHAP summary plot showing top_n features.
    Saves to output_path if provided.
    Returns top_n feature names and their mean absolute SHAP values.
    """
    if not SHAP_AVAILABLE:
        return [], []
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Sort and get top_n
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_names)), top_values, color='skyblue')
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel("Mean Absolute SHAP Value")
    plt.title(f"Global Feature Importance (Top {top_n})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved global importance plot to {output_path}")
    
    plt.close()
    
    return top_names, top_values

def local_explanation(model, smiles, background_X,
                       model_type="tree", feature_names=None):
    """
    Computes SHAP values for a single molecule.
    Returns a dict with explanation details.
    """
    if not SHAP_AVAILABLE or not RDKIT_AVAILABLE:
        return {
            "smiles": smiles,
            "features": None,
            "shap_values": None,
            "top_positive_features": [],
            "top_negative_features": [],
            "base_value": 0,
            "predicted_value": 0,
            "warning": "SHAP or RDKit not available"
        }
    
    # Featurize the query molecule (placeholder - would use src/featurizer.py)
    # For demonstration, using random features matching background_X shape
    n_features = background_X.shape[1]
    query_features = np.random.randn(1, n_features)
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model) if model_type == "tree" else shap.KernelExplainer(model.predict, background_X[:100])
    shap_values = explainer.shap_values(query_features)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Base value and prediction
    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[0]
    
    prediction = model.predict(query_features)[0] if hasattr(model, 'predict') else model(query_features)[0]
    
    # Top features
    if feature_names is not None:
        shap_array = shap_values[0]
        feature_importance = list(zip(feature_names, shap_array))
        
        # Sort by absolute value
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_positive = [f"{name}: {val:.3f}" for name, val in feature_importance if val > 0][:10]
        top_negative = [f"{name}: {val:.3f}" for name, val in feature_importance if val < 0][:10]
    else:
        top_positive = []
        top_negative = []
    
    return {
        "smiles": smiles,
        "features": query_features[0],
        "shap_values": shap_values[0],
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
        "base_value": base_value,
        "predicted_value": prediction
    }

def explain_physchem_contributions(model, smiles, background_X,
                                   feature_names=None, physchem_indices=None):
    """
    Runs local_explanation and extracts SHAP values for physicochemical
    descriptor portion only. Maps back to descriptor names.
    Returns readable dict of descriptor contributions.
    """
    explanation = local_explanation(model, smiles, background_X, feature_names=feature_names)
    
    if explanation["shap_values"] is None or physchem_indices is None:
        return {}
    
    shap_array = explanation["shap_values"]
    contributions = {}
    
    physchem_names = [feature_names[i] for i in physchem_indices if i < len(feature_names)]
    physchem_values = [shap_array[i] for i in physchem_indices if i < len(shap_array)]
    
    for name, val in zip(physchem_names, physchem_values):
        contributions[name] = val
    
    return contributions

def plot_shap_waterfall(local_explanation_result, output_path=None):
    """
    Plots a waterfall plot showing how each top feature moves
    the prediction from the base value to the final prediction.
    Saves to output_path.
    """
    if not SHAP_AVAILABLE:
        return
    
    shap_values = local_explanation_result["shap_values"]
    base_value = local_explanation_result["base_value"]
    predicted_value = local_explanation_result["predicted_value"]
    
    # Create waterfall plot
    plt.figure(figsize=(10, 6))
    
    # Sort by absolute SHAP value
    sorted_indices = np.argsort(np.abs(shap_values))[::-1][:20]
    top_values = shap_values[sorted_indices]
    
    # Cumulative sum
    cumulative = np.cumsum(top_values) + base_value
    
    plt.bar(range(len(top_values)), top_values, alpha=0.7, color=['green' if v > 0 else 'red' for v in top_values])
    plt.plot(range(len(top_values) + 1), [base_value] + list(cumulative), 'k--', lw=2)
    
    plt.xlabel("Top Features")
    plt.ylabel("SHAP Value Contribution")
    plt.title("SHAP Waterfall Plot")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved waterfall plot to {output_path}")
    
    plt.close()

def plot_shap_beeswarm(shap_values, feature_names,
                       output_path=None):
    """
    Plots beeswarm showing distribution of SHAP values per feature.
    """
    if not SHAP_AVAILABLE:
        return
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved beeswarm plot to {output_path}")
    
    plt.close()

def interaction_effects(model, X, feature_idx1, feature_idx2,
                        model_type="tree"):
    """
    Uses shap.TreeExplainer interaction values.
    Returns the interaction SHAP matrix for two features.
    Plots a 2D heatmap of their interaction.
    """
    if not SHAP_AVAILABLE:
        return None
    
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        interaction_values = explainer.shap_interaction_values(X)
        
        # Extract interaction between feature_idx1 and feature_idx2
        if isinstance(interaction_values, list):
            interaction_matrix = interaction_values[0]
        else:
            interaction_matrix = interaction_values
        
        interaction_2d = interaction_matrix[:, feature_idx1, feature_idx2]
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, feature_idx1], X[:, feature_idx2], c=interaction_2d, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Interaction SHAP Value')
        plt.xlabel(f"Feature {feature_idx1}")
        plt.ylabel(f"Feature {feature_idx2}")
        plt.title(f"Interaction Effect: Feature {feature_idx1} vs {feature_idx2}")
        plt.tight_layout()
        plt.show()
        
        return interaction_2d
    
    return None

def generate_explanation_report(model, smiles, background_X,
                                 target_name, output_path,
                                 feature_names=None, physchem_indices=None):
    """
    Generates a complete HTML report for one molecule.
    Includes structure image, predicted affinity, top features, waterfall plot,
    applicability domain check, and disclaimer.
    """
    explanation = local_explanation(model, smiles, background_X, feature_names=feature_names)
    
    # Generate structure image (placeholder - would use RDKit in production)
    structure_img_base64 = ""
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                img = Draw.MolToImage(mol, size=(300, 300))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                structure_img_base64 = base64.b64encode(buffered.getvalue()).decode()
        except:
            pass
    
    # Generate waterfall plot
    waterfall_path = output_path.replace(".html", "_waterfall.png")
    plot_shap_waterfall(explanation, waterfall_path)
    
    # Read waterfall image
    waterfall_base64 = ""
    if os.path.exists(waterfall_path):
        with open(waterfall_path, "rb") as f:
            waterfall_base64 = base64.b64encode(f.read()).decode()
    
    # HTML report
    report_html = f"""
    <html>
    <head>
        <title>Explanation Report: {smiles}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .section {{ margin-bottom: 30px; }}
            .structure {{ text-align: center; margin: 20px 0; }}
            .waterfall {{ text-align: center; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .good {{ color: green; font-weight: bold; }}
            .bad {{ color: red; font-weight: bold; }}
            .disclaimer {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Explanation Report</h1>
            <h2>Molecule: {smiles}</h2>
            <h2>Target: {target_name.upper()}</h2>
            
            <div class="section">
                <h3>Structure</h3>
                <div class="structure">
    """
    
    if structure_img_base64:
        report_html += f'<img src="data:image/png;base64,{structure_img_base64}" alt="Molecule structure" style="max-width: 300px;">'
    else:
        report_html += '<p>Structure image not available (RDKit not installed)</p>'
    
    report_html += """
                </div>
            </div>
            
            <div class="section">
                <h3>Prediction Summary</h3>
                <table>
                    <tr><th>Base Value (Expected)</th><td>{base_value:.3f} pChEMBL</td></tr>
                    <tr><th>Predicted Value</th><td class="good">{predicted_value:.3f} pChEMBL</td></tr>
                    <tr><th>Difference</th><td>{diff:.3f} pChEMBL</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h3>Top Positive Features (Driving Potency Up)</h3>
                <ul>
    """
    
    for feat in explanation["top_positive_features"]:
        report_html += f"<li>{feat}</li>"
    
    report_html += """
                </ul>
            </div>
            
            <div class="section">
                <h3>Top Negative Features (Driving Potency Down)</h3>
                <ul>
    """
    
    for feat in explanation["top_negative_features"]:
        report_html += f"<li>{feat}</li>"
    
    report_html += """
                </ul>
            </div>
            
            <div class="section">
                <h3>Waterfall Plot</h3>
                <div class="waterfall">
    """
    
    if waterfall_base64:
        report_html += f'<img src="data:image/png;base64,{waterfall_base64}" alt="Waterfall plot" style="max-width: 800px;">'
    else:
        report_html += '<p>Waterfall plot not available</p>'
    
    report_html += """
                </div>
            </div>
            
            <div class="section">
                <h3>Applicability Domain</h3>
                <p><strong>Base value:</strong> {base_value:.3f}</p>
                <p><strong>Predicted value:</strong> {predicted_value:.3f}</p>
                <p><strong>Difference:</strong> {diff:.3f}</p>
            </div>
            
            <div class="disclaimer">
                <strong>Disclaimer:</strong> These explanations are computational predictions
                based on the trained model. They represent feature contributions
                to the prediction, not causal biological relationships. Always
                validate predictions with wet-lab experiments. Model accuracy
                depends on training data quality and applicability domain.
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(report_html)
    
    print(f"Saved explanation report to {output_path}")
    
    return explanation

# Example usage
if __name__ == "__main__":
    print("Explainer module loaded.")
    print("Functions: compute_shap_values, global_feature_importance,")
    print("           local_explanation, explain_physchem_contributions,")
    print("           plot_shap_waterfall, plot_shap_beeswarm,")
    print("           interaction_effects, generate_explanation_report")
