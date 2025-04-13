import pandas as pd
from rdkit import Chem
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

def load_data(file_path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def convert_rdkit_to_smiles(rdkit_mol):
    """Convert RDKit molecule object to SMILES string."""
    return Chem.MolToSmiles(rdkit_mol)

def prepare_data(df):
    """Prepare the data for model training."""
    df = df[['SMILES', 'pIC50']].dropna()
    return df

def extract_embeddings(model_name, smiles_list):
    """Extract embeddings from the ChemBERTa model for a list of SMILES."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embeddings = []

    for smiles in smiles_list:
        inputs = tokenizer(smiles, return_tensors='pt')
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings

def evaluate_model(model, test_loader):
    """Evaluate the regression model using mean squared error."""
    from sklearn.metrics import mean_squared_error
    all_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            predictions = model(inputs).squeeze(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())

    mse = mean_squared_error(all_labels, all_preds)
    return mse
