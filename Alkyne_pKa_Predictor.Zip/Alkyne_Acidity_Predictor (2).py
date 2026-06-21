
import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import Descriptors

def train_production_engine():
    # Automatically trains the backend model using your locked training database
    log_file = "Alkyne_Project_Master_Log.xlsx"
    if not os.path.exists(log_file):
        raise FileNotFoundError("Missing master log file. Please ensure training data exists.")
    df = pd.read_excel(log_file)
    X_raw = df[["Feature_HOMO", "Feature_LUMO", "Feature_MolWt"]].values
    Y = df["Target_pKa"].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    model = Ridge(alpha=0.1)
    model.fit(X_scaled, Y)
    return model, scaler

def predict_single_alkyne(smiles, name="User_Molecule"):
    xtb_path = r"C:\Users\uwakm\xtb"
    model, scaler = train_production_engine()
    
    # Generate temporary structure natively
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return "Invalid SMILES String!"
    mol = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(mol, Chem.AllChem.ETKDGv3())
    Chem.AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94")
    
    xyz_path = "predict_temp.xyz"
    Chem.MolToXYZFile(mol, xyz_path)
    mw = Descriptors.MolWt(mol)
    
    # Background quantum execution
    command = f'"{xtb_path}" {xyz_path} --gfn 2'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    homo, lumo = -10.0, 0.0
    output_text = result.stdout if result.stdout else ""
    if os.path.exists("xtbout"):
        with open("xtbout", "r") as f: output_text += f.read()
        
    for line in output_text.split("\n"):
        if "(HOMO)" in line: homo = float(line.split()[-2])
        if "(LUMO)" in line: lumo = float(line.split()[-2])
            
    if os.path.exists(xyz_path): os.remove(xyz_path)
    for temp in ["xtbout", "xtbrestart", "wbo", "chrg", "coord", "charges", "gfnff_topo"]:
        if os.path.exists(temp): 
            try: os.remove(temp)
            except: pass
            
    # Standardize and project the final score
    raw_feat = np.array([[homo, lumo, mw]])
    scaled_feat = scaler.transform(raw_feat)
    return model.predict(scaled_feat).item()

# =============================================================================
# CHOOSE ANY NEW MOLECULE TO TEST YOUR INTERACTIVE ENGINE RIGHT HERE:
# =============================================================================
test_smiles = "O=C(N)C#C"  # Propynamide candidate
result_pka = predict_single_alkyne(test_smiles)
print(f"\n🔮 Interactive Engine Result:")
print(f"SMILES Input: {test_smiles} ➔ Predicted pKa: {result_pka:.2f}")
