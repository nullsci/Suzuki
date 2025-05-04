import pandas as pd
import numpy as np
import random
import re
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import rdBase
from multiprocessing import Pool, cpu_count # Import multiprocessing components
from tqdm import tqdm # Import tqdm for progress bar

# Suppress RDKit warnings (optional, but cleans up output)
rdBase.DisableLog('rdApp.warning')
# Completely suppress all RDKit logs to avoid UFF/MMFF warnings
import rdkit.RDLogger as rdl
rdl.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=UserWarning) # Suppress pandas warnings if any
warnings.filterwarnings("ignore", category=RuntimeWarning) # Suppress numpy NaN warnings

# --- Configuration ---
INPUT_CSV_FILE = 'suzuki.csv' # Ensure this file path is correct
OUTPUT_CSV_FILE = 'suzuki_rdkit_features.csv' # Changed output filename
DESCRIPTOR_NAMES = [
    'MolWt', 'NumHBA', 'NumHBD', 'MolLogP', 'Asphericity',
    'RadiusOfGyration', 'TPSA', 'NumRings', 'NumRotatableBonds', 'NumHeteroatoms'
]
# Note: Metal detection is now done with SMARTS patterns in the function

# --- RDKit Descriptor Calculation Function (improved with better metal handling) ---
def calculate_rdkit_descriptors(smiles_string):
    """
    Calculates 10 RDKit descriptors for a given SMILES string.
    Improved handling for metals and optimization failures.
    """
    if not smiles_string or pd.isna(smiles_string):
        return np.full(len(DESCRIPTOR_NAMES), np.nan)

    # Define improved metal detection
    def contains_metal_improved(mol):
        if mol is None:
            return False
        metal_patterns = [
            '[Fe,Pd,Pt,Ni,Rh,Ru,Ir,Au,Ag,Cu,Co,Mn,Mo,Cr,V,Ti,Zn]',  # Any metal
            '[Fe+,Fe++,Fe+++,Fe+2,Fe+3]',  # Iron with various charge notations
            '[Pd+,Pd++,Pd+2]',  # Palladium with charges
            # Add other metals with charges as needed
        ]
        for pattern in metal_patterns:
            try:
                patt = Chem.MolFromSmarts(pattern)
                if patt and mol.HasSubstructMatch(patt):
                    return True
            except:
                pass
        return False

    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return np.full(len(DESCRIPTOR_NAMES), np.nan)

        # Enhanced metal detection
        contains_metal = contains_metal_improved(mol)

        # For metal-containing compounds, skip 3D calculations
        if contains_metal:
            desc = [
                Descriptors.MolWt(mol), rdMolDescriptors.CalcNumHBA(mol), 
                rdMolDescriptors.CalcNumHBD(mol), Descriptors.MolLogP(mol), 
                np.nan, np.nan, Descriptors.TPSA(mol),
                rdMolDescriptors.CalcNumRings(mol), 
                rdMolDescriptors.CalcNumRotatableBonds(mol),
                rdMolDescriptors.CalcNumHeteroatoms(mol)
            ]
            return np.array(desc, dtype=float)

        # For non-metals, proceed with 3D conformer generation
        mol3d = Chem.AddHs(mol)
        embed_flag = AllChem.EmbedMolecule(mol3d, randomSeed=random.randint(1, 1000000))

        if embed_flag == -1:
            # If embedding fails, return 2D descriptors
            desc = [
                Descriptors.MolWt(mol), rdMolDescriptors.CalcNumHBA(mol), 
                rdMolDescriptors.CalcNumHBD(mol), Descriptors.MolLogP(mol), 
                np.nan, np.nan, Descriptors.TPSA(mol),
                rdMolDescriptors.CalcNumRings(mol), 
                rdMolDescriptors.CalcNumRotatableBonds(mol),
                rdMolDescriptors.CalcNumHeteroatoms(mol)
            ]
            return np.array(desc, dtype=float)

        # Improved force field optimization with fallbacks
        try:
            # Try MMFF
            props = AllChem.MMFFGetMoleculeProperties(mol3d)
            if props is None:
                # If MMFF fails, try UFF
                AllChem.UFFOptimizeMolecule(mol3d)
            else:
                AllChem.MMFFOptimizeMolecule(mol3d)
        except Exception:
            # If all optimizations fail, continue with unoptimized molecule
            pass

        # Calculate descriptors
        try:
            descriptors = [
                Descriptors.MolWt(mol3d), rdMolDescriptors.CalcNumHBA(mol3d), 
                rdMolDescriptors.CalcNumHBD(mol3d), Descriptors.MolLogP(mol3d), 
                rdMolDescriptors.CalcAsphericity(mol3d), 
                rdMolDescriptors.CalcRadiusOfGyration(mol3d),
                Descriptors.TPSA(mol3d), rdMolDescriptors.CalcNumRings(mol3d), 
                rdMolDescriptors.CalcNumRotatableBonds(mol3d),
                rdMolDescriptors.CalcNumHeteroatoms(mol3d)
            ]
            return np.array(descriptors, dtype=float)
        except Exception:
            return np.full(len(DESCRIPTOR_NAMES), np.nan)

    except Exception:
        return np.full(len(DESCRIPTOR_NAMES), np.nan)

# --- SMILES Splitting Function (remains the same) ---
def split_smiles(smiles_str, delimiters=['.', '~', ',']):
    """Splits a string containing multiple SMILES based on delimiters."""
    if not smiles_str or pd.isna(smiles_str):
        return []
    pattern = '|'.join(map(re.escape, delimiters))
    return [s.strip() for s in re.split(pattern, smiles_str) if s and s.strip()]

# --- Worker Function for Parallel Processing ---
def process_reaction_row(args):
    """
    Processes a single reaction's SMILES strings to calculate aggregated descriptors.

    Args:
        args (tuple): A tuple containing (reactants_smiles_str, additive_smiles_str,
                      solvent_smiles_str, yield_val).

    Returns:
        np.array: A numpy array containing the aggregated descriptors and yield.
    """
    reactants_smiles_str, additive_smiles_str, solvent_smiles_str, yield_val = args
    num_descriptors = len(DESCRIPTOR_NAMES) # Get length locally

    # Split SMILES
    reactant_list = split_smiles(reactants_smiles_str, delimiters=[','])
    additive_list = split_smiles(additive_smiles_str, delimiters=[',', '~'])
    solvent_list = split_smiles(solvent_smiles_str, delimiters=['.'])

    # Initialize summed descriptors
    summed_reactant_desc = np.zeros(num_descriptors)
    summed_additive_desc = np.zeros(num_descriptors)
    summed_solvent_desc = np.zeros(num_descriptors)

    # Process Reactants
    for smiles in reactant_list:
        desc = calculate_rdkit_descriptors(smiles)
        # nansum treats NaN as 0 for summation, which is the desired aggregation here
        summed_reactant_desc = np.nansum([summed_reactant_desc, desc], axis=0)

    # Process Additives
    for smiles in additive_list:
        desc = calculate_rdkit_descriptors(smiles)
        summed_additive_desc = np.nansum([summed_additive_desc, desc], axis=0)

    # Process Solvents
    for smiles in solvent_list:
        desc = calculate_rdkit_descriptors(smiles)
        summed_solvent_desc = np.nansum([summed_solvent_desc, desc], axis=0)

    # Combine aggregated features for the reaction row
    reaction_features = np.concatenate([
        summed_reactant_desc,
        summed_additive_desc,
        summed_solvent_desc,
        [yield_val] # Append the target variable (Yield)
    ])
    return reaction_features

# --- Main Processing Logic ---
if __name__ == "__main__": # IMPORTANT: Protect multiprocessing code execution

    print(f"Loading Suzuki reaction data from {INPUT_CSV_FILE}...")
    try:
        df = pd.read_csv(INPUT_CSV_FILE, sep=',') # Adjust separator if needed
    except FileNotFoundError:
        print(f"Error: File not found at {INPUT_CSV_FILE}")
        exit()
    except Exception as e_load:
        print(f"Error loading CSV: {e_load}. Check file path and separator.")
        exit()

    # Verify necessary columns
    required_cols = ['Reactants', 'Additive', 'Solvent', 'Yield']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input CSV must contain columns: {required_cols}")
        print(f"Found columns: {df.columns.tolist()}")
        exit()

    print(f"Preparing data for parallel processing ({len(df)} reactions)...")

    # Create a list of argument tuples for the worker function
    # Using .values is faster than iterrows for large dataframes
    tasks = [
        (row[0], row[1], row[2], row[3])
        for row in df[['Reactants', 'Additive', 'Solvent', 'Yield']].values
    ]

    # Determine number of workers (leave one core free for system stability)
    num_workers = max(1, cpu_count() - 1)
    # Alternatively, set a fixed number: num_workers = 4
    print(f"Starting parallel processing with {num_workers} workers...")

    all_features = []
    # Use multiprocessing Pool
    # `chunksize` can improve performance by reducing overhead for very fast tasks
    # Adjust chunksize based on dataset size and typical task duration (e.g., 50-200)
    chunk_size = max(1, len(df) // (num_workers * 4)) # Heuristic for chunk size

    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for potential performance gains if order doesn't matter
        # and wrap with tqdm for a progress bar. Use list() to collect all results.
        all_features = list(tqdm(
            pool.imap(process_reaction_row, tasks, chunksize=chunk_size),
            total=len(tasks),
            desc="Processing Reactions"
        ))

    print("Finished parallel processing.")
    # Note: The original 'error_count' tracking is not directly feasible here
    # as errors happen in separate processes. The use of np.nan provides the
    # mechanism for handling failures, consistent with the original script.

    print("Creating output DataFrame...")

    # --- Prepare Output DataFrame ---
    reactant_cols = [f'Reactant_Sum_{name}' for name in DESCRIPTOR_NAMES]
    additive_cols = [f'Additive_Sum_{name}' for name in DESCRIPTOR_NAMES]
    solvent_cols = [f'Solvent_Sum_{name}' for name in DESCRIPTOR_NAMES]
    output_columns = reactant_cols + additive_cols + solvent_cols + ['Yield']

    output_df = pd.DataFrame(all_features, columns=output_columns)

    # Optional: Check for rows that might have failed entirely (all NaNs except maybe yield)
    # This doesn't count individual SMILES errors but flags problematic rows.
    nan_rows = output_df.drop('Yield', axis=1).isnull().all(axis=1).sum()
    if nan_rows > 0:
        print(f"Warning: {nan_rows} rows resulted in all NaN features (excluding yield).")

    print(f"Saving aggregated RDKit features and yield to {OUTPUT_CSV_FILE}...")
    output_df.to_csv(OUTPUT_CSV_FILE, index=False)

    print("Done.")