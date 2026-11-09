from pathlib import Path

import torch as t
from os import PathLike
import time

from cordconv2d.model.model import Iwmymdosr

class EarlyStopping:
    def __init__(self, model: Iwmymdosr, model_dir: PathLike[str] | str, model_prefix = time.time(), best_only = True, patience = 5, min_delta = 1e-4):
        self.best_only = best_only
        self.model_prefix = model_prefix
        self.model_dir = Path(model_dir)
        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.prepare()

    def prepare(self):
        self.min_loss = float("inf")
        self.patience_counter = 0
        self.imrovements_count = 0
        self.best = None
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def step(self, loss):
        if self.improved(loss+self.min_delta):
            self.imrovements_count += 1
            self.patience_counter = 0
            self.min_loss = loss
            self.save_model()
        else:        
            self.patience_counter += 1
            if self.patience_counter == self.patience:
                if self.improved(loss):
                    self.imrovements_count += 1
                    self.save_model()
                return True

        return False
        
    def improved(self, loss):
        return loss < self.min_loss
    
    def save_model(self):
        self.best = self.model.state_dict()
        if self.best_only:
            t.save(self.best, self.get_model_output_path(self.get_model_best_name()))
        else:
            t.save(self.best, self.get_model_output_path(self.get_model_output_name()))
            
    def get_model_best_name(self):
        return f"{self.model.__class__.__name__}_{self.model_prefix}_best.pt"
    
    def get_model_output_path(self, name: str):
        return f"{self.model_dir}/{name}"
    
    def get_model_output_name(self):
        return f"{self.model.__class__.__name__}_{self.model_prefix}_{self.imrovements_count}.pt"