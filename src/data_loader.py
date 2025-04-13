import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import Dataset

def load_data(file_path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def convert_rdkit_to_smiles(rdkit_mol):
    """Convert RDKit molecule to SMILES string."""
    return Chem.MolToSmiles(rdkit_mol)

def prepare_data(df):
    """Prepare the data for model training."""
    #df['mol'] = df['mol'].str[1:-1]
    #df['mol'] = df['mol'].apply(Chem.MolFromSmiles)
    # print 5 lines of mol column
    print("preparing data")
    print(df['mol'].apply(type).unique())  # ici on voit que le type de la colonne mol est str et non rdkit
    print(df['mol'].head()) 
    df['mol'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x) if isinstance(x, str) else x) # Convertir les chaînes SMILES en objets moléculaires RDKit.
    print(df['mol'].apply(type).unique())  # Doit afficher uniquement <class 'rdkit.Chem.rdchem.Mol'>
 
    df['SMILES'] = df['mol'].apply(convert_rdkit_to_smiles)
    print("\nle nombre de val nan avant le drop:", df.isna().sum())
    df = df[['SMILES', 'pIC50']].dropna()
    print("\nle nbr de val nan apres le dop", df.isna().sum())
    print ("end of preparing data")
    return df

class SMILESDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        smiles = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        inputs = self.tokenizer(smiles, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        return inputs, torch.tensor(label, dtype=torch.float)