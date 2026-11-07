"""
Comprehensive Molecular Featurizer

Every featurization method explained:

- Morgan fingerprints: circular fingerprints that encode local atomic environments
  around each atom up to a certain radius. They capture substructures like "this carbon
  is connected to two oxygens and one nitrogen" which directly relates to binding.
  The 2048-bit vector represents presence/absence of these substructures.

- MACCS keys: 166 predefined structural keys defined by MDL. Each bit represents
  a specific chemical feature (like "contains a benzene ring" or "has a sulfur atom").
  They're interpretable because each bit maps to a known chemical property.

- RDKit fingerprints: path-based fingerprints that encode linear sequences of atoms
  and bonds. Different from Morgan because they capture longer-range connectivity
  patterns rather than just circular neighborhoods.

- Mordred descriptors: 1600+ molecular descriptors covering topology (how atoms
  are connected), geometry (3D shape), electronics (charge distribution), and
  pharmacophore features (hydrogen bond donors/acceptors). They give a complete
  numerical profile of the molecule.

- Physicochemical descriptors: hand-crafted Lipinski-era features like molecular
  weight, LogP (lipophilicity), hydrogen bond donors/acceptors, topological polar
  surface area, etc. These directly relate to drug-likeness and ADMET properties.

- Why combining multiple representations outperforms any single one:
  Each method captures different aspects. Morgan sees local substructures,
  MACCS sees predefined features, RDKit sees paths, Mordred sees global properties,
  and physchem sees drug-like characteristics. Together they give a complete picture.

- Applicability domain: predictions for molecules very different from the training
  set are unreliable. We measure this by computing similarity to training molecules.
  If the query molecule is too different (low similarity), the prediction should
  be treated with caution.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# Try to import RDKit and related libraries
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import Draw
    from rdkit.Chem.Fingerprints import FingerprintMols
    RDkit_AVAILABLE = True
except ImportError:
    RDkit_AVAILABLE = False
    print("WARNING: RDKit not available in sandbox. Will use fallback implementations.")

# Try to import Mordred
try:
    from mordred import Calculator, descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False
    print("WARNING: Mordred not available in sandbox. Will use fallback.")

# Try to import molvs
try:
    from molvs import standardize_smiles
    MOLVS_AVAILABLE = True
except ImportError:
    MOLVS_AVAILABLE = False
    print("WARNING: MolVS not available in sandbox. Will use basic standardization.")

def smiles_to_mol(smiles):
    """
    Returns RDKit mol object or None.
    Sanitizes with Chem.SanitizeMol and strips salts using molvs.standardize_smiles first.
    """
    if not RDkit_AVAILABLE:
        return None
    
    try:
        # Standardize with molvs if available
        if MOLVS_AVAILABLE:
            try:
                smiles = standardize_smiles(str(smiles))
            except:
                pass
        
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        print(f"Error converting SMILES {smiles}: {e}")
        return None

def morgan_fingerprint(mol, radius=2, n_bits=2048):
    """
    Returns numpy array of length 2048.
    Uses GetMorganFingerprintAsBitVect to encode circular substructures.
    """
    if not RDkit_AVAILABLE or mol is None:
        return np.zeros(n_bits)
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def morgan_fingerprint_counts(mol, radius=2, n_bits=2048):
    """
    Returns count-based Morgan fingerprint not just bits.
    Captures how many times each substructure appears, not just presence/absence.
    """
    if not RDkit_AVAILABLE or mol is None:
        return np.zeros(n_bits)
    
    fp = AllChem.GetMorganFingerprint(mol, radius)
    arr = np.zeros(n_bits)
    for idx, count in fp.GetNonzeroElements().items():
        if idx < n_bits:
            arr[idx] = count
    return arr

def maccs_keys(mol):
    """
    Returns numpy array of length 166.
    Uses MACCSkeys.GenMACCSKeys to encode predefined structural features.
    """
    if not RDkit_AVAILABLE or mol is None:
        return np.zeros(166)
    
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def rdkit_fingerprint(mol, n_bits=2048):
    """
    Returns numpy array of length 2048.
    Uses RDKFingerprint to encode path-based connectivity patterns.
    """
    if not RDkit_AVAILABLE or mol is None:
        return np.zeros(n_bits)
    
    fp = FingerprintMols.FingerprintMol(mol, minPath=1, maxPath=7, fpSize=n_bits)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def physicochemical_descriptors(mol):
    """
    Returns numpy array of physicochemical descriptors unnormalized:
    MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA,
    NumRotatableBonds, RingCount, NumAromaticRings,
    FractionCSP3, NumHeteroatoms, NumStereocenters,
    MaxPartialCharge, MinPartialCharge,
    BalabanJ, BertzCT, Chi0, Chi1, Kappa1, Kappa2, Kappa3,
    LabuteASA, PEOE_VSA1 through PEOE_VSA14,
    SlogP_VSA1 through SlogP_VSA12
    
    Replaces NaN and Inf with 0.
    """
    if not RDkit_AVAILABLE or mol is None:
        # Return 48 zeros for the 48 descriptors
        return np.zeros(48)
    
    try:
        # Basic descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        rotb = Descriptors.NumRotatableBonds(mol)
        rings = Descriptors.RingCount(mol)
        arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        hetero = Descriptors.NumHeteroatoms(mol)
        stereo = Descriptors.NumStereocenters(mol)
        
        # Partial charges
        try:
            charges = [atom.GetPartialCharge() for atom in mol.GetAtoms()]
            max_charge = max(charges) if charges else 0
            min_charge = min(charges) if charges else 0
        except:
            max_charge, min_charge = 0, 0
        
        # Topological descriptors
        balaban = Descriptors.BalabanJ(mol)
        bertz = Descriptors.BertzCT(mol)
        chi0 = Descriptors.Chi0(mol)
        chi1 = Descriptors.Chi1(mol)
        kappa1 = Descriptors.Kappa1(mol)
        kappa2 = Descriptors.Kappa2(mol)
        kappa3 = Descriptors.Kappa3(mol)
        
        # Surface area and VSA
        labute = rdMolDescriptors.CalcLabuteASA(mol)
        
        # PEOE_VSA (14 descriptors)
        peoe_vsa = []
        try:
            peoe_vsa = rdMolDescriptors.CalcPEOE_VSA(mol)
            if len(peoe_vsa) < 14:
                peoe_vsa = list(peoe_vsa) + [0] * (14 - len(peoe_vsa))
            else:
                peoe_vsa = peoe_vsa[:14]
        except:
            peoe_vsa = [0] * 14
        
        # SlogP_VSA (12 descriptors)
        slogp_vsa = []
        try:
            slogp_vsa = rdMolDescriptors.CalcSlogP_VSA(mol)
            if len(slogp_vsa) < 12:
                slogp_vsa = list(slogp_vsa) + [0] * (12 - len(slogp_vsa))
            else:
                slogp_vsa = slogp_vsa[:12]
        except:
            slogp_vsa = [0] * 12
        
        # Combine all
        descriptors_list = [
            mw, logp, hbd, hba, tpsa, rotb, rings, arom_rings,
            csp3, hetero, stereo, max_charge, min_charge,
            balaban, bertz, chi0, chi1, kappa1, kappa2, kappa3,
            labute
        ] + list(peoe_vsa) + list(slogp_vsa)
        
        # Replace NaN and Inf with 0
        arr = np.array(descriptors_list, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        return arr
    except Exception as e:
        print(f"Error computing physchem descriptors: {e}")
        return np.zeros(48)

def mordred_descriptors(mol):
    """
    Uses mordred.Calculator with mordred.descriptors.
    Returns numpy array, replaces NaN/errors with 0.
    Returns None if calculation fails entirely.
    """
    if not MORDRED_AVAILABLE or mol is None:
        return None
    
    try:
        calc = Calculator(descriptors, ignore_3D=True)
        res = calc(mol)
        arr = np.array(res, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    except Exception as e:
        print(f"Mordred calculation failed: {e}")
        return None

def lipinski_violations(mol):
    """
    Returns count of Lipinski rule of five violations:
      MW > 500, LogP > 5, HBD > 5, HBA > 10
    And Veber rule violations:
      RotBonds > 10, TPSA > 140
    """
    if not RDkit_AVAILABLE or mol is None:
        return {"lipinski": 0, "veber": 0, "total": 0}
    
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        rotb = Descriptors.NumRotatableBonds(mol)
        
        lipinski = 0
        if mw > 500: lipinski += 1
        if logp > 5: lipinski += 1
        if hbd > 5: lipinski += 1
        if hba > 10: lipinski += 1
        
        veber = 0
        if rotb > 10: veber += 1
        if tpsa > 140: veber += 1
        
        return {"lipinski": lipinski, "veber": veber, "total": lipinski + veber}
    except:
        return {"lipinski": 0, "veber": 0, "total": 0}

def featurize_smiles(smiles, method="combined"):
    """
    Featurizes a single SMILES string.
    method can be:
      "morgan": only Morgan 2048 bit
      "morgan_counts": only count Morgan
      "maccs": only MACCS 166 bit
      "rdkit": only RDKit fingerprint
      "physchem": only physicochemical descriptors
      "morgan_physchem": Morgan + physchem concatenated
      "all_fingerprints": Morgan + MACCS + RDKit concatenated
      "combined": Morgan + MACCS + RDKit + physchem concatenated
      "mordred": Mordred descriptors only
      "full": everything concatenated
    
    Returns numpy array or None.
    Logs method and output dimensions on first call.
    """
    mol = smiles_to_mol(smiles)
    
    if method == "morgan":
        return morgan_fingerprint(mol)
    elif method == "morgan_counts":
        return morgan_fingerprint_counts(mol)
    elif method == "maccs":
        return maccs_keys(mol)
    elif method == "rdkit":
        return rdkit_fingerprint(mol)
    elif method == "physchem":
        return physicochemical_descriptors(mol)
    elif method == "morgan_physchem":
        morgan = morgan_fingerprint(mol)
        physchem = physicochemical_descriptors(mol)
        return np.concatenate([morgan, physchem])
    elif method == "all_fingerprints":
        morgan = morgan_fingerprint(mol)
        maccs = maccs_keys(mol)
        rdkit = rdkit_fingerprint(mol)
        return np.concatenate([morgan, maccs, rdkit])
    elif method == "combined":
        morgan = morgan_fingerprint(mol)
        maccs = maccs_keys(mol)
        rdkit = rdkit_fingerprint(mol)
        physchem = physicochemical_descriptors(mol)
        return np.concatenate([morgan, maccs, rdkit, physchem])
    elif method == "mordred":
        return mordred_descriptors(mol)
    elif method == "full":
        morgan = morgan_fingerprint(mol)
        maccs = maccs_keys(mol)
        rdkit = rdkit_fingerprint(mol)
        physchem = physicochemical_descriptors(mol)
        mordred = mordred_descriptors(mol)
        
        parts = [morgan, maccs, rdkit, physchem]
        if mordred is not None:
            parts.append(mordred)
        
        return np.concatenate(parts)
    else:
        raise ValueError(f"Unknown method: {method}")

def featurize_dataframe(df, smiles_col="canonical_smiles",
                        method="combined", n_jobs=-1):
    """
    Featurizes all SMILES in a dataframe in parallel using joblib.
    Shows tqdm progress bar.
    Returns df with features column added.
    Reports success and failure counts.
    """
    smiles_list = df[smiles_col].tolist()
    
    print(f"Featurizing {len(smiles_list)} compounds with method '{method}'...")
    
    # Parallel featurization
    results = Parallel(n_jobs=n_jobs)(
        delayed(featurize_smiles)(smiles, method) for smiles in tqdm(smiles_list)
    )
    
    # Count successes and failures
    success_count = sum(1 for r in results if r is not None)
    failure_count = len(results) - success_count
    
    print(f"  Success: {success_count}, Failure: {failure_count}")
    
    # Add features column (None for failures)
    df["features"] = results
    
    return df

def compute_tanimoto_similarity(smiles1, smiles2):
    """
    Returns Tanimoto similarity between two Morgan fingerprints.
    Uses DataStructs.TanimotoSimilarity.
    """
    if not RDkit_AVAILABLE:
        return 0.0
    
    mol1 = smiles_to_mol(smiles1)
    mol2 = smiles_to_mol(smiles2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def compute_similarity_matrix(smiles_list):
    """
    Returns NxN numpy matrix of pairwise Tanimoto similarities.
    Shows progress bar.
    """
    n = len(smiles_list)
    matrix = np.zeros((n, n))
    
    print(f"Computing {n}x{n} similarity matrix...")
    
    for i in tqdm(range(n), desc="Computing similarities"):
        for j in range(i, n):
            sim = compute_tanimoto_similarity(smiles_list[i], smiles_list[j])
            matrix[i, j] = sim
            matrix[j, i] = sim
    
    return matrix

def get_murcko_scaffold(smiles):
    """
    Returns the Murcko scaffold SMILES of a molecule.
    Returns None if invalid.
    """
    if not RDkit_AVAILABLE:
        return None
    
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None

def check_applicability_domain(query_smiles, training_smiles_list,
                               threshold=0.4):
    """
    Computes Tanimoto similarity of query to all training molecules.
    Returns:
      max_similarity: float
      mean_similarity: float
      in_domain: True if max_similarity >= threshold
      warning: message if out of domain
    """
    if not RDkit_AVAILABLE:
        return {
            "max_similarity": 0.0,
            "mean_similarity": 0.0,
            "in_domain": False,
            "warning": "RDKit not available, cannot compute similarity"
        }
    
    similarities = []
    for train_smiles in training_smiles_list:
        sim = compute_tanimoto_similarity(query_smiles, train_smiles)
        similarities.append(sim)
    
    if not similarities:
        return {
            "max_similarity": 0.0,
            "mean_similarity": 0.0,
            "in_domain": False,
            "warning": "No training molecules provided"
        }
    
    max_sim = max(similarities)
    mean_sim = sum(similarities) / len(similarities)
    in_domain = max_sim >= threshold
    
    warning = ""
    if not in_domain:
        warning = f"Query molecule is outside applicability domain (max similarity {max_sim:.3f} < {threshold})"
    
    return {
        "max_similarity": max_sim,
        "mean_similarity": mean_sim,
        "in_domain": in_domain,
        "warning": warning
    }

# Example usage comment
if __name__ == "__main__":
    print("Featurizer module loaded.")
    print("Methods available: morgan, morgan_counts, maccs, rdkit, physchem,")
    print("                   morgan_physchem, all_fingerprints, combined, mordred, full")
