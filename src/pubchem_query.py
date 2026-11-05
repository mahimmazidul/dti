"""
PubChem is a free database run by the National Center for Biotechnology Information that stores information about chemical compounds, their structures, and how they behave in biological tests. Anyone can search it to find details about drugs, chemicals, and their properties without paying for access.

SMILES notation is a way to write down the chemical structure of a molecule using plain text characters instead of drawing it. It uses letters and symbols to show which atoms are connected and how, making it easy for computers to read and store chemical information while still representing the actual shape of the molecule.
"""

import pubchempy as pcp
import time

def get_compound_properties(drug_name):
    """
    Takes a drug name
    Uses pubchempy to fetch the compound
    Returns a dict with: cid, molecular_formula, molecular_weight, iupac_name, xlogp, canonical_smiles, inchikey
    If not found, return None with a printed message
    """
    try:
        compounds = pcp.get_compounds(drug_name, 'name')
        if not compounds:
            print(f"No compound found with name '{drug_name}' in PubChem")
            return None
            
        compound = compounds[0]
        return {
            "cid": compound.cid,
            "molecular_formula": compound.molecular_formula,
            "molecular_weight": compound.molecular_weight,
            "iupac_name": compound.iupac_name,
            "xlogp": compound.xlogp,
            "canonical_smiles": compound.canonical_smiles,
            "inchikey": compound.inchikey
        }
    except Exception as e:
        print(f"Error fetching compound properties for '{drug_name}': {e}")
        return None

def get_compound_image_url(cid):
    """
    Takes a PubChem CID integer
    Returns the direct URL to the 2D structure PNG image
    Format: https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG
    """
    if cid:
        return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
    return None
