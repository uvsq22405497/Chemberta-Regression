import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from data_loader import load_data, prepare_data
from utils_functions import extract_embeddings, evaluate_model
from model import SimpleRegressor
import numpy as np

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Print CUDA device name
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Specify the model name
model_name = "seyonec/ChemBERTa-zinc-base-v1"

# Load the tokenizer and the pretrained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
chemberta_model = AutoModel.from_pretrained(model_name).to(device)

# Load the dataset
file_path = 'c:/Users/benal/OneDrive/Bureau/ChemBerta/SMILES_Big_Data_Set.csv'
df = load_data(file_path)

# Vérifiez les types de données dans le DataFrame
print(df.dtypes)

# Afficher les premières lignes du DataFrame
print(df.head())

# Afficher taille du DataFrame
print(df.shape)


# Filtrer les colonnes numériques
numeric_df = df.select_dtypes(include=[np.number])

# Afficher head du DataFrame numérique
print(numeric_df.head())
# Afficher taille du DataFrame numérique
print(numeric_df.shape)

# Vérifiez les valeurs NaN et infinies
print("Nombre de valeurs NaN dans le dataset :", df.isna().sum().sum())
print("Nombre de valeurs infinies dans le dataset :", np.isinf(numeric_df.values).sum())

# Afficher le nombre de valeurs NaN par colonne
print(df.isna().sum())

# Remplacer les valeurs NaN par la moyenne des colonnes
numeric_df = numeric_df.fillna(numeric_df.mean())
# Afficher le nombre de valeurs NaN par colonne
print(numeric_df.isna().sum())

# Vérifiez à nouveau les valeurs NaN après remplacement
print("Nombre de valeurs NaN après remplacement :", numeric_df.isna().sum().sum())

# Replace the original numeric columns in the DataFrame with the processed numeric_df
df[numeric_df.columns] = numeric_df

# Take the first row of df and assign it to df
df = df.iloc[:1]

# Prepare the data
df = prepare_data(df)