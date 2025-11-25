import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MTSDataset(Dataset):
    def __init__(self, dataframe, target_col=None):
        """
        Args:
            dataframe (pd.DataFrame): El dataframe ja agregat (1 fila per usuari).
            target_col (str): Nom de la columna objectiu ('age' o 'is_male').
                              Si és None, estem en mode inferència (test).
        """
        self.df = dataframe
        self.target_col = target_col
        
        # Seleccionem només les columnes numèriques per simplificar el Baseline
        # (Més endavant aquí tractarem les categòriques amb embeddings)
        self.features = self.df.select_dtypes(include=[np.number]).drop(
            columns=['user_id', 'is_male', 'age'], errors='ignore'
        )
        
        # Normalització bàsica (molt important per xarxes neuronals)
        self.features = (self.features - self.features.mean()) / (self.features.std() + 1e-6)
        
        # Convertim a matriu de numpy per velocitat
        self.X = self.features.values.astype(np.float32)
        
        # Preparem les etiquetes (Labels) si en tenim
        self.y = None
        if target_col and target_col in dataframe.columns:
            self.y = dataframe[target_col].values.astype(np.float32) # Float per regressió (edat) o BCE (gènere)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retorna un parell (features, label)
        features = torch.tensor(self.X[idx])
        
        if self.y is not None:
            label = torch.tensor(self.y[idx])
            return features, label
        else:
            return features