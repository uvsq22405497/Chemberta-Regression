import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class ChemBERTaRegressor(nn.Module):
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1"):
        super(ChemBERTaRegressor, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chemberta = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.chemberta.config.hidden_size, 1)

    def forward(self, smiles):
        inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = self.chemberta(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]
        prediction = self.regressor(cls_token)
        return prediction.squeeze()

class SimpleRegressor(nn.Module):
    def __init__(self, input_dim):
        super(SimpleRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x