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
import wandb
import os
from sklearn.metrics import r2_score

learning_rate = 0.001
nb_epochs = 20

# Initialize Weights & Biases
# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    # Set the wandb project where this run will be logged.
    project="Chemberta-Regression",
    # Track hyperparameters and run metadata.
    name="ChemBERTa-Regression-Run - lr "+ str(learning_rate),
    config={
        "learning_rate": learning_rate,
        "architecture": "SimpleRegressor",
        "dataset": "SMILES_Big_Data_Set",
        "epochs": nb_epochs,
    },
)

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
print ("\nle nombre de valeurs NaN par colonne avant remplacement dans df :")
print(df.isna().sum())

# Remplacer les valeurs NaN par la moyenne des colonnes
numeric_df = numeric_df.fillna(numeric_df.mean())
# Afficher le nombre de valeurs NaN par colonne
print("\nle nombre de valeurs NaN par colonne après remplacement dans numeric_df :")
print(numeric_df.isna().sum())

# Vérifiez à nouveau les valeurs NaN après remplacement
print("Nombre de valeurs NaN après remplacement :", numeric_df.isna().sum().sum())

# Replace the original numeric columns in the DataFrame with the processed numeric_df
df[numeric_df.columns] = numeric_df

print("\nle nombre de valeurs NaN par colonne après remplacement dans df :")
print(df.isna().sum())

# Prepare the data
df = prepare_data(df)

print("\nretour au train apres prepare_data")

# Extract ChemBERTa embeddings
smiles_list = df['SMILES'].tolist()

if os.path.exists("chemberta_embeddings.csv"):
    print("Chargement des embeddings depuis le fichier CSV...")
    df_with_smiles = pd.read_csv("chemberta_embeddings.csv")
    smiles = df_with_smiles['SMILES']
    pIC50 = df_with_smiles['pIC50']
    embeddings = df_with_smiles.iloc[:, 2:].values
else:
    print("Génération des embeddings...")
    embeddings = extract_embeddings(model_name, smiles_list)
    df_embeddings = pd.DataFrame(embeddings)
    df_with_smiles = pd.concat([df[['SMILES', 'pIC50']], df_embeddings], axis=1)
    df_with_smiles.to_csv("chemberta_embeddings.csv", index=False)

# Sauvegarder les embeddings
df_embeddings = pd.DataFrame(embeddings)
df_with_smiles = pd.concat([df[['SMILES', 'pIC50']], df_embeddings], axis=1)
df_with_smiles.to_csv("chemberta_embeddings.csv", index=False)

# Vérifiez les valeurs NaN et infinies dans les embeddings
print("Nombre de valeurs NaN dans les embeddings :", np.isnan(embeddings).sum())
print("Nombre de valeurs infinies dans les embeddings :", np.isinf(embeddings).sum())

# Remplacer les valeurs NaN dans les embeddings par la moyenne des colonnes
embeddings = np.nan_to_num(embeddings, nan=np.nanmean(embeddings))

# Sauvegarder les embeddings dans un fichier CSV
df_embeddings = pd.DataFrame(embeddings)  # Convertir les embeddings en DataFrame
df_with_smiles = pd.concat([df[['SMILES', 'pIC50']], df_embeddings], axis=1)  # Ajouter SMILES et pIC50
df_with_smiles.to_csv("chemberta_embeddings.csv", index=False)  # Sauvegarder dans un fichier CSV
print("Embeddings sauvegardés dans chemberta_embeddings.csv")

# Charger les embeddings depuis le fichier CSV
df_with_smiles = pd.read_csv("chemberta_embeddings.csv")  # Charger le fichier CSV
smiles = df_with_smiles['SMILES']  # Extraire la colonne SMILES
pIC50 = df_with_smiles['pIC50']  # Extraire la colonne pIC50
embeddings = df_with_smiles.iloc[:, 2:].values  # Extraire les colonnes des embeddings
print("Embeddings chargés depuis chemberta_embeddings.csv")

# Add embeddings to the DataFrame
df_embeddings = pd.DataFrame(embeddings)
df_final = pd.concat([df, df_embeddings], axis=1)

# Define X (features) and y (target)
X = df_final.iloc[:, -768:].values  # sélectionne les 768 dernières colonnes du DataFrame, qui correspondent aux dimensions des embeddings ChemBERTa.
#.values convertit le DataFrame pandas en une matrice NumPy, qui est plus adaptée pour l'entraînement de modèles machine learning.
#X est une matrice NumPy où chaque ligne représente les 768 dimensions des embeddings pour une molécule.

y = df_final["pIC50"].values# Extraire la colonne pIC50 comme cible (target) pour le modèle.
#"df_final["pIC50"] sélectionne la colonne pIC50 du DataFrame, qui contient les valeurs cibles pour la régression ".values" convertit cette colonne pandas en un tableau NumPy.
#y est un tableau NumPy contenant les valeurs cibles associées à chaque molécule.

# Vérifiez les valeurs NaN et infinies dans X et y
print("Nombre de valeurs NaN dans X :", np.isnan(X).sum())
print("Nombre de valeurs infinies dans X :", np.isinf(X).sum())
print("Nombre de valeurs NaN dans y :", np.isnan(y).sum())
print("Nombre de valeurs infinies dans y :", np.isinf(y).sum())

# Remplacer les valeurs NaN dans X par la moyenne des colonnes
X = np.nan_to_num(X, nan=np.nanmean(X))

# Remplacer les valeurs NaN dans y par la moyenne
y = np.nan_to_num(y, nan=np.nanmean(y))

# Vérifiez à nouveau les valeurs NaN et infinies dans X et y après remplacement
print("Nombre de valeurs NaN dans X après remplacement :", np.isnan(X).sum())
print("Nombre de valeurs infinies dans X après remplacement :", np.isinf(X).sum())
print("Nombre de valeurs NaN dans y après remplacement :", np.isnan(y).sum())
print("Nombre de valeurs infinies dans y après remplacement :", np.isinf(y).sum())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors and move to GPU
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
#Ce segment de code est une étape essentielle pour préparer les données d'entraînement et de test dans un format (tensor) compatible avec PyTorch, tout en exploitant les capacités matérielles disponibles (GPU si possible).
#tensor est un type de données utilisé par PyTorch pour représenter des tableaux multidimensionnels, similaires aux tableaux NumPy, mais optimisés pour le calcul sur GPU.


# Vérifiez les valeurs NaN et infinies dans les tenseurs
print("Nombre de valeurs NaN dans X_train :", torch.isnan(X_train).sum().item())
print("Nombre de valeurs infinies dans X_train :", torch.isinf(X_train).sum().item())
print("Nombre de valeurs NaN dans y_train :", torch.isnan(y_train).sum().item())
print("Nombre de valeurs infinies dans y_train :", torch.isinf(y_train).sum().item())



# Create DataLoader

# TensorDataset est une classe de PyTorch qui permet de créer un ensemble de données à partir de tenseurs.
# Elle est souvent utilisée pour regrouper les entrées (features) et les étiquettes (labels) dans un format compatible avec DataLoader.
train_dataset = TensorDataset(X_train, y_train)# creation des tenseurs d'entraînement
test_dataset = TensorDataset(X_test, y_test)# creation des tenseurs de test
# DataLoader est une classe de PyTorch qui permet de charger les données en mini-lots (batches) pour l'entraînement et l'évaluation des modèles.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#train_loader et test_loader sont des objets DataLoader qui permettent de charger les données d'entraînement et de test en mini-lots de 32 échantillons chacun.
#train_loader utilise l'option shuffle=True pour mélanger les données à chaque époque, ce qui est important pour l'entraînement des modèles de machine learning.    

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]# on recupere le nombre de colonnes de X_train, qui correspond à la taille de l'embedding ChemBERTa (768 dimensions).
model = SimpleRegressor(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)# on utilise Adam comme optimiseur, qui est un algorithme d'optimisation populaire pour l'entraînement des modèles de deep learning.
#Adam est un optimiseur adaptatif qui ajuste le taux d'apprentissage pour chaque paramètre du modèle en fonction de la moyenne des gradients passés.

# Training loop
num_epochs = nb_epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Log training loss
    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss}")
    run.log({"train_loss": train_loss})  # Log training loss to wandb

    # Evaluation on the test set
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = model(inputs)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
        
        # Convert predictions and true values to NumPy arrays
        y_pred = np.array(y_pred).flatten()
        y_true = np.array(y_true).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)  # R² as a proxy for precision in regression
        print(f"Epoch {epoch+1}/{num_epochs}, Test MSE: {mse}, Test R²: {r2}")
        run.log({"test_mse": mse, "test_r2": r2})  # Log metrics to wandb

run.finish()  # Finish the wandb run