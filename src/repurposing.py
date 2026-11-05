"""
Drug repurposing is the strategy of finding new therapeutic uses for existing approved drugs. It is faster and cheaper than developing new drugs from scratch because approved drugs already have safety data and known manufacturing processes, meaning a drug that took 10 years and 2 billion dollars to approve can potentially be redirected to a new disease in a fraction of that time.

Famous real examples include Sildenafil, which was originally developed for hypertension but became Viagra for erectile dysfunction, Thalidomide which was withdrawn for birth defects but later approved for leprosy and multiple myeloma, and Metformin which is currently being studied for anti-aging effects.

ChEMBL enables computational repurposing by mapping approved drugs to all their known targets, allowing researchers to identify cases where a drug approved for disease A also hits a protein relevant to disease B. This computational approach provides starting points for research, though it is important to remember that these predictions require experimental validation and are not clinical conclusions on their own.
"""

from src.chembl_query import search_target, get_bioactivities, search_molecule
import time

def get_phase_label(phase):
    """
    Converts a clinical trial phase number into a readable string description.
    Helps researchers quickly understand the regulatory status of a drug candidate.
    """
    if phase == 4:
        return "Approved"
    elif phase == 3:
        return "Phase 3 Clinical Trial"
    elif phase == 2:
        return "Phase 2 Clinical Trial"
    elif phase == 1:
        return "Phase 1 Clinical Trial"
    elif phase == 0:
        return "Preclinical"
    else:
        return "Unknown"

def find_repurposing_candidates(target_name, min_phase=2, organism="Homo sapiens"):
    """
    Searches for approved or clinical-stage drugs that interact with a specific protein target.
    Filters by minimum clinical trial phase to ensure only drugs with real evidence are returned,
    helping identify potential new uses for existing medications.
    """
    # 1. Find the target
    targets = search_target(target_name, organism=organism)
    if not targets:
        print(f"No target found for '{target_name}'")
        return []
        
    first_target = targets[0]
    target_id = first_target.get("target_chembl_id")
    
    # 2. Get bioactivities
    bioactivities = get_bioactivities(target_id, limit=50)
    
    candidates = []
    checked = 0
    
    # 3. Check each molecule
    for bio in bioactivities:
        mol_id = bio.get("molecule_chembl_id")
        if not mol_id:
            continue
            
        # Search molecule details
        mol_info = search_molecule(mol_id)
        time.sleep(0.3)  # Respect rate limit
        checked += 1
        
        if checked % 10 == 0:
            print(f"Checked {checked} molecules so far...")
            
        max_phase = mol_info.get("max_phase")
        if max_phase is not None and max_phase >= min_phase:
            candidates.append({
                "molecule_chembl_id": mol_id,
                "pref_name": mol_info.get("pref_name", mol_id),
                "max_phase": max_phase,
                "pchembl_value": bio.get("pchembl_value"),
                "ic50_nM": bio.get("standard_value"),
                "smiles": bio.get("canonical_smiles"),
                "target_name": target_name,
                "target_chembl_id": target_id
            })
            
    return candidates

def format_repurposing_results(candidates):
    """
    Formats the list of repurposing candidates into a readable table.
    Sorts by clinical phase first and then by binding strength, making it easy
    to see which approved drugs have the strongest evidence for the target.
    """
    if not candidates:
        return "No approved or clinical stage drugs found for this target."
        
    # Sort by max_phase descending then pchembl_value descending
    sorted_candidates = sorted(
        candidates,
        key=lambda x: (x.get("max_phase") or 0, x.get("pchembl_value") or 0),
        reverse=True
    )
    
    from tabulate import tabulate
    headers = ["Drug Name", "Phase", "pChEMBL", "IC50 (nM)", "ChEMBL ID"]
    table_data = []
    
    for c in sorted_candidates:
        table_data.append([
            c.get("pref_name", "Unknown")[:20] + "...",
            get_phase_label(c.get("max_phase")),
            c.get("pchembl_value", "N/A"),
            c.get("ic50_nM", "N/A"),
            c.get("molecule_chembl_id", "N/A")
        ])
        
    return tabulate(table_data, headers=headers, tablefmt="grid")
