import torch
import numpy as np
from typing import List
from pathlib import Path
from .jtnn import JTNNVAE, Vocab


MODEL_PATH = str(Path(__file__).parent.resolve()) + '/models/model.iter-4'
VOCAB_PATH = str(Path(__file__).parent.resolve()) + '/data/vocab.txt'

class JTNNEmbed:
    """
    Handles JTNN latent vector generation. These vectors can be used as molecular descriptors.
    """

    def __init__(
        self,
        smiles: List[str],
        batch_size: int = 100,
        hidden_size: int = 450,
        latent_size: int = 56,
        depth: int = 3,
        vocab_path: str = VOCAB_PATH,
        model_path: str = MODEL_PATH,
    ) -> None:
        """
        Args:
            smiles (List[str]): List of smiles for which to calculate features.
            batch_size (int, optional): Defaults to 100.
            hidden_size (int, optional): Defaults to 450.
            latent_size (int, optional): Defaults to 56.
            depth (int, optional): Defaults to 3.
            vocab_path (str, optional): Defaults to VOCAB_PATH.
            model_path (str, optional): Defaults to MODEL_PATH.
        """

        self.smiles = smiles
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        self.vocab_path = vocab_path
        self.model_path = model_path

        vocab = Vocab([x.strip("\r\n ") for x in open(self.vocab_path)])

        self.model = JTNNVAE(vocab, self.hidden_size, self.latent_size, self.depth)
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model = self.model.cpu()

    def get_features(self) -> np.ndarray:
        """
        Calculate JTNN embeddings for supplied moelcules.

        Returns:
            np.ndarray: Array of features for input molecules of dimension (n_smiles x n_features).
        """
        features = []
        batch_size = self.batch_size

        for i in range(0, len(self.smiles), batch_size):
            batch = self.smiles[i:i + batch_size]
            mol_vec = self.model.encode_latent_mean(batch)
            features.append(mol_vec.data.cpu().numpy())

        return features
