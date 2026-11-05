"""
A drug-target interaction network is a visual map where the dots represent either drugs or proteins, and the lines connecting them show which drugs interact with which proteins. The thickness or darkness of the lines shows how strongly the drug binds, helping us see the relationships at a glance.

Visualizing these networks matters in drug discovery because many approved drugs actually hit multiple different targets in the body, a concept called polypharmacology. Seeing these connections helps scientists spot opportunities to use existing drugs for new diseases and understand why some drugs cause side effects by hitting unintended targets.

In these networks, IC50 and pChEMBL values tell us about binding strength visually. A lower IC50 means the drug binds more tightly, so the lines from stronger interactions appear thicker or darker to show they are more potent.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def build_interaction_network(target_name, bioactivities, output_file=None):
    """
    Builds a visual network graph showing which drugs interact with a protein target.
    Creates nodes for the target and drugs, connects them with lines whose thickness
    shows binding strength, and either displays the graph or saves it to a file.
    """
    # 1. Create a networkx Graph
    G = nx.Graph()
    
    # 2. Add the target as a node
    G.add_node(target_name, node_type="target", label=target_name)
    
    # 3. Add drug nodes and edges
    for bio in bioactivities:
        standard_value = bio.get("standard_value")
        molecule_id = bio.get("molecule_chembl_id")
        pchembl = bio.get("pchembl_value")
        
        # Skip entries where standard_value or molecule_id is None
        if standard_value is None or molecule_id is None:
            continue
            
        # Add drug node
        G.add_node(molecule_id, node_type="drug", label=molecule_id)
        
        # Add edge with weight and ic50
        weight = pchembl if pchembl is not None else 1.0
        G.add_edge(target_name, molecule_id, weight=weight, ic50=standard_value)
        
    # 4. Use spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # 5. Draw the graph
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    ax.set_facecolor('#f0f0f0')  # Light grey background
    
    # Separate nodes
    target_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "target"]
    drug_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]
    
    # Draw target node (red, size 2000)
    nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_color='red', node_size=2000)
    
    # Draw drug nodes (steelblue, size 800)
    nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_color='steelblue', node_size=800)
    
    # Calculate edge widths
    edges = G.edges(data=True)
    widths = []
    for u, v, data in edges:
        pchembl = data.get("weight", 1.0)
        # Formula: width = (pchembl_value / 10) * 4, clamped between 0.5 and 4.0
        width = (pchembl / 10.0) * 4.0
        width = max(0.5, min(4.0, width))
        widths.append(width)
        
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=widths, edge_color='gray', alpha=0.7)
    
    # Draw labels
    # Target node: font size 11 bold
    target_labels = {n: d.get("label") for n, d in G.nodes(data=True) if d.get("node_type") == "target"}
    nx.draw_networkx_labels(G, pos, labels=target_labels, font_size=11, font_weight='bold')
    
    # Drug nodes: font size 7
    drug_labels = {n: d.get("label") for n, d in G.nodes(data=True) if d.get("node_type") == "drug"}
    nx.draw_networkx_labels(G, pos, labels=drug_labels, font_size=7)
    
    # Title and subtitle
    plt.title(f"Drug-Target Interaction Network: {target_name}", fontsize=14)
    plt.suptitle("Edge width proportional to pChEMBL value (binding potency)", fontsize=10, y=0.95)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Protein target',
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Drug compound',
               markerfacecolor='steelblue', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # 6. Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Graph saved to {output_file}")
    else:
        plt.show()
        
    # 8. Close figure
    plt.close()

def summarize_network(target_name, bioactivities):
    """
    Creates a simple summary of the drug-target network without drawing it.
    Counts how many drugs interact, finds the strongest and weakest binders,
    and calculates average binding strength values.
    """
    valid_bio = [b for b in bioactivities if b.get("standard_value") is not None and b.get("molecule_chembl_id") is not None]
    
    total_drugs = len(valid_bio)
    
    if not valid_bio:
        return {
            "target": target_name,
            "total_drugs": 0,
            "most_potent_drug": None,
            "least_potent_drug": None,
            "average_pchembl": 0.0,
            "average_ic50_nM": 0.0
        }
        
    # Sort by pchembl_value descending
    sorted_by_pchembl = sorted(valid_bio, key=lambda x: x.get("pchembl_value") or 0, reverse=True)
    most_potent = sorted_by_pchembl[0].get("molecule_chembl_id")
    least_potent = sorted_by_pchembl[-1].get("molecule_chembl_id")
    
    # Calculate averages
    pchembl_values = [b.get("pchembl_value") for b in valid_bio if b.get("pchembl_value") is not None]
    ic50_values = [float(b.get("standard_value")) for b in valid_bio if b.get("standard_value") is not None]
    
    avg_pchembl = round(sum(pchembl_values) / len(pchembl_values), 2) if pchembl_values else 0.0
    avg_ic50 = round(sum(ic50_values) / len(ic50_values), 2) if ic50_values else 0.0
    
    return {
        "target": target_name,
        "total_drugs": total_drugs,
        "most_potent_drug": most_potent,
        "least_potent_drug": least_potent,
        "average_pchembl": avg_pchembl,
        "average_ic50_nM": avg_ic50
    }
