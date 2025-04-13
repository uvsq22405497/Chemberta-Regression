# ChemBERTa Regression Project

This project implements a regression model using ChemBERTa to predict pIC50 values from molecular representations. The model leverages embeddings extracted from molecular structures represented in SMILES format.

## Project Structure

```
chemberta-regression
├── data
│   └── SMILES_Big_Data_Set.csv  # Dataset containing molecular representations and properties
├── src
│   ├── __init__.py               # Marks the directory as a Python package
│   ├── data_loader.py             # Functions to load and preprocess the dataset
│   ├── model.py                   # Defines the regression model using ChemBERTa
│   ├── train.py                   # Contains the training loop for the model
│   └── utils.py                   # Utility functions for data processing and evaluation
├── requirements.txt               # Lists project dependencies
└── README.md                      # Documentation for the project
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Load the Data**: Use the `data_loader.py` to load the dataset and convert RDKit objects to SMILES.
2. **Train the Model**: Run the `train.py` script to train the regression model on the dataset.
3. **Evaluate the Model**: The training script will also evaluate the model's performance using metrics like mean squared error.

## Dependencies

This project requires the following Python packages:

- pandas
- RDKit
- transformers
- torch
- scikit-learn

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
