from rdkit import Chem
import pandas as pd
import os

# --- Helper Function for Canonicalization ---
def safe_canonicalize(smi):
    """
    Safely canonicalizes a SMILES string.
    Returns canonical SMILES if successful, otherwise the original SMILES
    (or empty string if input is empty/falsy). Prints a warning on failure.
    """
    if not smi:
        return '' # Return empty string for empty/None input
    mol = Chem.MolFromSmiles(smi)
    if mol:
        return Chem.MolToSmiles(mol)
    else:
        print(f"Warning: RDKit could not parse SMILES: '{smi}'. Using raw string.")
        return smi # Return the original if parsing fails

# --- Data Dictionaries (Keep as is) ---
reactant_1_smiles = {
    '6-chloroquinoline': 'C1=C(Cl)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC',
    '6-Bromoquinoline': 'C1=C(Br)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC',
    '6-triflatequinoline': 'C1C2C(=NC=CC=2)C=CC=1OS(C(F)(F)F)(=O)=O.CCC1=CC(=CC=C1)CC',
    '6-Iodoquinoline': 'C1=C(I)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC',
    '6-quinoline-boronic acid hydrochloride': 'C1C(B(O)O)=CC=C2N=CC=CC=12.Cl.O',
    'Potassium quinoline-6-trifluoroborate': '[B-](C1=CC2=C(C=C1)N=CC=C2)(F)(F)F.[K+].O',
    '6-Quinolineboronic acid pinacol ester': 'B1(OC(C(O1)(C)C)(C)C)C2=CC3=C(C=C2)N=CC=C3.O'
}

reactant_2_smiles = {
    '2a, Boronic Acid': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B(O)O',
    '2b, Boronic Ester': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B4OC(C)(C)C(C)(C)O4',
    '2c, Trifluoroborate': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1[B-](F)(F)F.[K+]',
    '2d, Bromide': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1Br'
}

catalyst_smiles = {
    'Pd(OAc)2': 'CC(=O)O~CC(=O)O~[Pd]'
}

ligand_smiles = {
    'P(tBu)3': 'CC(C)(C)P(C(C)(C)C)C(C)(C)C',
    'P(Ph)3 ': 'c3c(P(c1ccccc1)c2ccccc2)cccc3',
    'AmPhos': 'CC(C)(C)P(C1=CC=C(C=C1)N(C)C)C(C)(C)C',
    'P(Cy)3': 'C1(CCCCC1)P(C2CCCCC2)C3CCCCC3',
    'P(o-Tol)3': 'CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C',
    'CataCXium A': 'CCCCP(C12CC3CC(C1)CC(C3)C2)C45CC6CC(C4)CC(C6)C5',
    'SPhos': 'COc1cccc(c1c2ccccc2P(C3CCCCC3)C4CCCCC4)OC',
    'dtbpf': 'CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.[Fe]',
    'XPhos': 'P(c2ccccc2c1c(cc(cc1C(C)C)C(C)C)C(C)C)(C3CCCCC3)C4CCCCC4',
    'dppf': 'C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.[Fe+2]',
    'Xantphos': 'O6c1c(cccc1P(c2ccccc2)c3ccccc3)C(c7cccc(P(c4ccccc4)c5ccccc5)c67)(C)C',
    'None': ''
}

reagent_1_smiles = {
    'NaOH': '[OH-].[Na+]',
    'NaHCO3': '[Na+].OC([O-])=O',
    'CsF': '[F-].[Cs+]',
    'K3PO4': '[K+].[K+].[K+].[O-]P([O-])([O-])=O',
    'KOH': '[K+].[OH-]',
    'LiOtBu': '[Li+].[O-]C(C)(C)C',
    'Et3N': 'CCN(CC)CC',
    'None': ''
}

solvent_1_smiles = {
    'MeCN': 'CC#N.O',
    'THF': 'C1CCOC1.O',
    'DMF': 'CN(C)C=O.O',
    'MeOH': 'CO.O',
    'MeOH/H2O_V2 9:1': 'CO.O', # Note: Both MeOH entries map to the same simplified SMILES
    'THF_V2': 'C1CCOC1.O'   # Note: Same as THF
}
# --- End Data Dictionaries ---

# --- Define Fixed Product SMILES and Canonicalize it Once ---
product_smi_raw = 'C1=C(C2=C(C)C=CC3N(C4OCCCC4)N=CC2=3)C=CC2=NC=CC=C12'
mol_product = Chem.MolFromSmiles(product_smi_raw)
if mol_product is None:
    print(f"CRITICAL ERROR: RDKit could not parse the fixed product SMILES: '{product_smi_raw}'. Check the string and RDKit installation.")
    exit() # Stop execution if product SMILES is invalid
else:
    canonical_product_smi = Chem.MolToSmiles(mol_product)
    print(f"Using canonical product SMILES: {canonical_product_smi}")
# --- End Product Definition ---


# --- Function to process a row and return structured data ---
def process_reaction_row(row):
    """
    Processes a row from the input DataFrame and returns a dictionary
    containing structured reaction components (Reactants, Products, Additive, Solvent, Yield).
    Applies canonicalization to individual components.
    """
    # 1. Reactants
    r1_name = row.get('Reactant_1_Name', None) # Use .get for safety
    r2_name = row.get('Reactant_2_Name', None)
    r1_smi = reactant_1_smiles.get(r1_name, '') if r1_name else ''
    r2_smi = reactant_2_smiles.get(r2_name, '') if r2_name else ''
    # Canonicalize each reactant and filter out empty strings
    reactants_list = [safe_canonicalize(smi) for smi in [r1_smi, r2_smi] if smi]
    reactants_str = ','.join(reactants_list)

    # 2. Product (already defined and canonicalized)
    products_str = canonical_product_smi # Use the pre-canonicalized version

    # 3. Additives (Catalyst + Ligand + Reagent)
    cat_name = row.get('Catalyst_1_Short_Hand', None)
    lig_name = row.get('Ligand_Short_Hand', None)
    rea_name = row.get('Reagent_1_Short_Hand', None)
    cat_smi = catalyst_smiles.get(cat_name, '') if cat_name else ''
    lig_smi = ligand_smiles.get(lig_name, '') if lig_name else ''
    rea_smi = reagent_1_smiles.get(rea_name, '') if rea_name else ''
    # Canonicalize each additive and filter out empty strings
    additives_list = [safe_canonicalize(smi) for smi in [cat_smi, lig_smi, rea_smi] if smi]
    additives_str = ','.join(additives_list)

    # 4. Solvent
    sol_name = row.get('Solvent_1_Short_Hand', None)
    sol_smi = solvent_1_smiles.get(sol_name, '') if sol_name else ''
    # Canonicalize the solvent
    solvent_str = safe_canonicalize(sol_smi)

    # 5. Yield
    yield_pct = row.get('Product_Yield_PCT_Area_UV', 0.0) # Default to 0.0 if missing
    yield_fraction = yield_pct / 100.0

    # Return as a dictionary (which .apply will turn into a Series)
    return {
        'Reactants': reactants_str,
        'Products': products_str,
        'Additive': additives_str,
        'Solvent': solvent_str,
        'Yield': yield_fraction
    }
# --- End Function ---


# --- Main Script Logic ---
# Define input and output file paths
data_dir = './'
excel_file_path = os.path.join(data_dir, 'aap9112_data_file_s1.xlsx')
# Choose a descriptive name for the structured output
output_csv_path = os.path.join(data_dir, 'suzuki.csv')

# Load data
print(f"Loading data from: {excel_file_path}")
try:
    df = pd.read_excel(excel_file_path)
    print(f"Successfully loaded {len(df)} rows.")
    # Optional: Check if necessary columns exist
    required_cols = ['Reactant_1_Name', 'Reactant_2_Name', 'Catalyst_1_Short_Hand',
                     'Ligand_Short_Hand', 'Reagent_1_Short_Hand', 'Solvent_1_Short_Hand',
                     'Product_Yield_PCT_Area_UV']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in Excel file: {missing_cols}")
        exit()

except FileNotFoundError:
    print(f"Error: Input Excel file not found at {excel_file_path}")
    exit() # Stop execution if file is missing
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit()

# Process rows to create the structured data
print("Processing rows to generate structured reaction data...")
# Use apply to create the new columns based on the process_reaction_row function
processed_data = df.apply(process_reaction_row, axis=1)

# Convert the result of apply (which is a Series of dictionaries) into a DataFrame
reactions_df = pd.DataFrame(processed_data.tolist())
print("Reaction data processed.")

# --- Save the structured DataFrame to a single CSV file ---
print(f"Saving {len(reactions_df)} processed reactions to: {output_csv_path}")

try:
    # Save to CSV format, using comma as separator, without the DataFrame index
    reactions_df.to_csv(output_csv_path, sep=',', index=False, quoting=1) # quoting=1 ensures strings with commas are quoted
    print("Successfully saved data to CSV.")
except Exception as e:
    print(f"Error saving data to CSV: {e}")

# Display the head of the final DataFrame as confirmation
print("\nFirst 5 rows of the generated structured data:")
print(reactions_df.head())

print("\nScript finished.") 