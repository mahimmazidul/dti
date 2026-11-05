"""
ChEMBL is a large database that collects information about chemical compounds and how they interact with proteins and other biological targets in living things. It is maintained by researchers at the European Bioinformatics Institute and provides free access to data about drug-like molecules, their chemical structures, and how strongly they bind to targets.

IC50 is a measurement that tells us how much of a drug is needed to reduce a biological activity by half. A lower IC50 value means the drug is more potent because less of it is needed to achieve the effect, which is important when comparing how strong different drugs are against the same target.

A drug target in biological terms is a specific molecule in the body, usually a protein or gene, that a drug is designed to interact with. When the drug binds to this target, it can change how the target works, which is how the drug produces its effect in treating a disease.
"""

from chembl_webresource_client.new_client import new_client
import sys

def search_target(target_name, organism="Homo sapiens"):
    """
    Takes a protein/gene name like "EGFR" or "BRCA1"
    Queries ChEMBL target endpoint using chembl_webresource_client
    Filters by organism
    Returns a list of dicts with: target_chembl_id, pref_name, target_type, organism
    If nothing found, return empty list
    """
    try:
        target = new_client.target
        results = target.filter(pref_name__icontains=target_name, organism=organism)
        
        target_list = []
        for res in results:
            target_list.append({
                "target_chembl_id": res.get("target_chembl_id"),
                "pref_name": res.get("pref_name"),
                "target_type": res.get("target_type"),
                "organism": res.get("organism")
            })
            # Limit to reasonable number to avoid excessive data
            if len(target_list) >= 10:
                break
                
        return target_list
    except Exception as e:
        print(f"Error searching target '{target_name}': {e}")
        return []

def get_bioactivities(target_chembl_id, limit=20):
    """
    Takes a ChEMBL target ID like "CHEMBL203"
    Queries the activity endpoint
    Filters for standard_type IC50
    Returns a list of dicts with: molecule_chembl_id, canonical_smiles, standard_value, standard_units, pchembl_value, assay_chembl_id
    Sort results by pchembl_value descending (most potent first)
    Skip entries where standard_value is None
    """
    try:
        activity = new_client.activity
        results = activity.filter(
            target_chembl_id=target_chembl_id,
            standard_type="IC50"
        )
        
        bioactivities = []
        for res in results:
            standard_value = res.get("standard_value")
            if standard_value is None:
                continue
                
            bioactivities.append({
                "molecule_chembl_id": res.get("molecule_chembl_id"),
                "canonical_smiles": res.get("canonical_smiles"),
                "standard_value": standard_value,
                "standard_units": res.get("standard_units"),
                "pchembl_value": res.get("pchembl_value"),
                "assay_chembl_id": res.get("assay_chembl_id")
            })
            
            if len(bioactivities) >= limit:
                break
                
        # Sort results by pchembl_value descending (most potent first)
        # Handle cases where pchembl_value might be None
        bioactivities.sort(
            key=lambda x: x.get("pchembl_value") or 0, 
            reverse=True
        )
        
        return bioactivities
    except Exception as e:
        print(f"Error getting bioactivities for target '{target_chembl_id}': {e}")
        return []

def search_molecule(drug_name):
    """
    Takes a drug name like "Imatinib" or "Aspirin"
    Queries ChEMBL molecule endpoint by preferred name
    Returns a dict with: molecule_chembl_id, pref_name, max_phase, molecule_type, smiles
    """
    try:
        molecule = new_client.molecule
        results = molecule.filter(pref_name__icontains=drug_name)
        
        for res in results:
            return {
                "molecule_chembl_id": res.get("molecule_chembl_id"),
                "pref_name": res.get("pref_name"),
                "max_phase": res.get("max_phase"),
                "molecule_type": res.get("molecule_type"),
                "smiles": res.get("smiles")
            }
            
        print(f"No molecule found with name '{drug_name}' in ChEMBL")
        return {}
    except Exception as e:
        print(f"Error searching molecule '{drug_name}': {e}")
        return {}
