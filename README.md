***ChemBERTa Regression***
Ce projet implémente un pipeline de régression chimique basé sur ChemBERTa, un modèle pré-entraîné pour le traitement des représentations moléculaires (SMILES). L'objectif est de prédire des propriétés chimiques (comme le pIC50) à partir de chaînes SMILES en utilisant des embeddings générés par ChemBERTa et un modèle de régression personnalisé.

**Fonctionnalités principales**
Extraction des embeddings ChemBERTa : Utilisation du modèle pré-entraîné seyonec/ChemBERTa-zinc-base-v1 pour générer des représentations vectorielles (768 dimensions) des molécules.
Préparation des données : Nettoyage des données, gestion des valeurs manquantes et conversion des SMILES en objets RDKit.
Entraînement d'un modèle de régression : Utilisation d'un modèle PyTorch personnalisé (SimpleRegressor) pour prédire les propriétés chimiques.
Évaluation des performances : Calcul des métriques comme le Mean Squared Error (MSE) et le R² (coefficient de détermination) pour évaluer la qualité des prédictions.
Support GPU : Accélération des calculs grâce à l'utilisation de CUDA si un GPU est disponible.
Suivi des expériences : Intégration avec Weights & Biases (wandb) pour suivre les pertes, les métriques et les hyperparamètres.
**Structure du projet**
chemberta-regression/
│
├── src/
│   ├── train.py               # Script principal pour l'entraînement et l'évaluation
│   ├── data_loader.py         # Chargement et préparation des données
│   ├── model.py               # Définition du modèle de régression
│   ├── utils_functions.py     # Fonctions utilitaires (extraction des embeddings, évaluation)
│
├── SMILES_Big_Data_Set.csv    # Jeu de données contenant les SMILES et les propriétés chimiques
├── requirements.txt           # Liste des dépendances Python
├── README.md                  # Documentation du projet
**Dépendances**
Les principales bibliothèques utilisées dans ce projet incluent :

PyTorch : Pour l'entraînement du modèle de régression.
Transformers : Pour charger le modèle ChemBERTa.
RDKit : Pour manipuler les molécules chimiques.
pandas et NumPy : Pour la manipulation des données.
scikit-learn : Pour le calcul des métriques (MSE, R²).
Weights & Biases (wandb) : Pour le suivi des expériences.
Installez toutes les dépendances avec :
pip install -r requirements.txt
Utilisation
Préparer les données :
Placez votre fichier CSV contenant les SMILES et les propriétés chimiques dans le dossier racine.
Assurez-vous que le fichier est correctement formaté (colonnes SMILES et pIC50).
Exécuter le script d'entraînement :
python src/train.py
Suivre les performances :
Les métriques d'entraînement et de test seront affichées dans la console.
Les résultats seront également enregistrés dans Weights & Biases si configuré.
Résultats
Les embeddings ChemBERTa sont sauvegardés dans un fichier CSV (chemberta_embeddings.csv) pour éviter de les recalculer à chaque exécution.
Les métriques comme le MSE et le R² sont calculées à chaque époque pour suivre les performances du modèle.
Améliorations possibles
Ajouter d'autres métriques comme le MAE (Mean Absolute Error).
Tester différents modèles de régression (par exemple, réseaux de neurones plus complexes).
Optimiser les hyperparamètres (taux d'apprentissage, taille des lots, etc.) avec des outils comme wandb sweep.
Auteur
Ce projet a été développé pour explorer l'utilisation de modèles pré-entraînés comme ChemBERTa dans des tâches de régression chimique. N'hésitez pas à contribuer ou à poser des questions !

Vous pouvez personnaliser cette description en fonction de vos besoins spécifiques.
