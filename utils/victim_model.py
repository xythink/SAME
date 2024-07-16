import torch
import numpy as np
from typing import List, Tuple
import torch.nn as nn
from ml_common import get_device


class Victim_Model(object):
    def __init__(
        self,
        origin_model: nn.Module,
        bounds: Tuple[float, float]=[-1, 1],
        num_classes: int = 10,
    ):
        self.device = get_device()
        self.origin_model = origin_model
        self.bounds = bounds
        self.num_classes = num_classes
        self.n_queries = 0
        

    def eval(self):
        self.origin_model.eval()

    def to(self, device: str):
        self.origin_model.to(device)
        return self


    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        """ clamp """
        x = torch.clamp(x, self.bounds[0], self.bounds[1])
        return x

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """ clamp and quantize the input """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
        if len(x.size()) in [1, 3]:  # assuming a vector/image as input
            x = x.unsqueeze(0)
        x = self.clamp(x)

        return x

    def __call__(
        self, x: torch.Tensor, label: bool = False) -> torch.Tensor:
        '''
            Feature:
            1. input preprocess;
            2. query cost log;
            3. support diverse model prediction fusion (hash model);
        '''
        x = self.preprocess(x)
        self.n_queries += x.shape[0]
        out = self.origin_model(x)

        if label:
            out = torch.argmax(out, dim=-1)
        return out
    
    def get_layer_embeddings(self, x):
        x = self.preprocess(x)
        out, layer_embeddings = self.origin_model.forward_with_EE(x)
        return out, layer_embeddings


    def get_n_queries(self) -> int:
        return self.n_queries
