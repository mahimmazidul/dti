from src.chembl_query import search_target, get_bioactivities, search_molecule
from src.pubchem_query import get_compound_properties, get_compound_image_url
from src.openfda_query import get_drug_label_summary
from tabulate import tabulate
import time

def analyze(query, mode="target"):
    """
    Main logic that combines ChEMBL and PubChem.
    
    mode="target": query is a protein name, find drugs that hit it
    mode="drug": query is a drug name, find its targets and properties
    """
    if mode == "target":
        # 1. Call search_target from chembl_query
        targets = search_target(query)
        if not targets:
            print(f"No target found for '{query}'")
            return {"mode": "target", "query": query, "results": []}
            
        # 2. Take the first result
        first_target = targets[0]
        target_id = first_target.get("target_chembl_id")
        
        # 3. Call get_bioactivities with that target ID
        bioactivities = get_bioactivities(target_id, limit=20)
        
        # 4. For each active molecule, optionally call get_compound_properties 
        #    from pubchem_query (only for top 5 to avoid rate limits)
        results = []
        for i, bio in enumerate(bioactivities):
            mol_id = bio.get("molecule_chembl_id")
            mol_name = mol_id  # Fallback to ID if no name
            
            # Get PubChem properties for top 5 only
            pubchem_props = None
            if i < 5:
                # We'll use the molecule_chembl_id or try to infer name if possible
                # For simplicity in this first module, we'll use the ID as query
                # In a full implementation, we might need to map ChEMBL ID to name first
                pubchem_props = get_compound_properties(mol_id)
                if pubchem_props:
                    mol_name = pubchem_props.get("iupac_name") or mol_id
                # Respect rate limits
                time.sleep(1)
                
            results.append({
                "drug_name": mol_name,
                "ic50_nm": bio.get("standard_value"),
                "pchembl": bio.get("pchembl_value"),
                "smiles": bio.get("canonical_smiles"),
                "pubchem_props": pubchem_props
            })
            
        return {
            "mode": "target",
            "query": query,
            "target_info": first_target,
            "results": results
        }
        
    elif mode == "drug":
        # 1. Call search_molecule from chembl_query
        chembl_info = search_molecule(query)
        
        # 2. Call get_compound_properties from pubchem_query
        pubchem_info = get_compound_properties(query)
        
        # Respect rate limits if we were doing loops (not needed here but good practice)
        time.sleep(1)
        
        # 3. Call get_drug_label_summary for FDA safety data
        fda_summary = get_drug_label_summary(query)
        
        # 4. Combine and return
        return {
            "mode": "drug",
            "query": query,
            "chembl_info": chembl_info,
            "pubchem_info": pubchem_info,
            "fda": fda_summary,
            "results": [chembl_info, pubchem_info] if chembl_info or pubchem_info else []
        }
    
    return {"mode": mode, "query": query, "results": []}

def format_results(results, mode="target"):
    """
    Formats the results dict into a readable table using tabulate
    mode="target": show columns drug name, IC50 (nM), pChEMBL, SMILES
    mode="drug": show molecular weight, formula, XLogP, SMILES, max clinical phase
    """
    if not results or "results" not in results or not results["results"]:
        return "No results found."
        
    data = results["results"]
    
    if mode == "target":
        headers = ["Drug Name", "IC50 (nM)", "pChEMBL", "SMILES"]
        table_data = []
        for item in data:
            table_data.append([
                item.get("drug_name", "N/A")[:20] + "...",
                item.get("ic50_nm", "N/A"),
                item.get("pchembl", "N/A"),
                item.get("smiles", "N/A")[:30] + "..."
            ])
        return tabulate(table_data, headers=headers, tablefmt="grid")
        
    elif mode == "drug":
        headers = ["Molecular Weight", "Formula", "XLogP", "SMILES", "Max Phase"]
        table_data = []
        
        # Combine info from both sources
        chembl = results.get("chembl_info", {})
        pubchem = results.get("pubchem_info", {})
        
        table_data.append([
            pubchem.get("molecular_weight", "N/A") if pubchem else "N/A",
            pubchem.get("molecular_formula", "N/A") if pubchem else "N/A",
            pubchem.get("xlogp", "N/A") if pubchem else "N/A",
            pubchem.get("canonical_smiles", chembl.get("smiles", "N/A"))[:30] + "...",
            chembl.get("max_phase", "N/A")
        ])
        
        return tabulate(table_data, headers=headers, tablefmt="grid")
        
    return ""
