import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from ml_models import get_model
from ml_datasets import get_dataloaders, ds_choices, nclasses_dict
from ml_common import train, get_device, test, calculate_execution_time
from utils import Victim_Model
from tqdm import tqdm


import os

class Base_Defender():
    '''
    Workflow:
    1. Init Defender
    2. Train/load victim model
    3. Init defense strategy
    4. query.
    '''
    def __init__(
        self,
        exp_id='Alpha',
        model='conv3',
        dataset='mnist',
        dataset_ood='kmnist',
        batch_size=128,
        lr=0.1,
        epochs=50,
        augment=True,
        defense_name='base',
    ):
        self.exp_id = exp_id
        self.model_name = model
        self.dataset = dataset
        self.dataset_ood = dataset_ood
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.defense_name = defense_name
        self.augment = augment
        self.origin_victim_model = None
        self.origin_victim_model_file_name = None
        self.device = get_device()
    
    def train_victim(self):

        dataloader_train, dataloader_test = get_dataloaders(self.dataset, self.batch_size, augment=self.augment)

        self.victim_save_path = f"./exp/{self.dataset}/{self.exp_id}/{self.defense_name}/victim/"
        if not os.path.exists(self.victim_save_path):
            os.makedirs(self.victim_save_path)
        model_file_name = f"{self.victim_save_path}/{self.model_name}-{self.epochs}.pt"
        criterion = nn.CrossEntropyLoss()
        print(f"\n==Training Victim Model==")

        model = get_model(self.model_name, self.dataset, pretrained=False)
        model = model.to(self.device)

        if os.path.exists(model_file_name):
            print("Victim model already exists. Load from file:",model_file_name)
            model.load_state_dict(torch.load(model_file_name))
            return model, model_file_name

        opt = SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        sch = CosineAnnealingLR(opt, self.epochs, last_epoch=-1)

        train(
            model,
            dataloader_train,
            dataloader_test,
            criterion,
            opt,
            self.epochs,
            sch,
            disable_pbar=False,
        )
        torch.save(model.state_dict(), model_file_name)
        self.origin_victim_model = model
        self.origin_victim_model_file_name = model_file_name
        return model, model_file_name
    
    @calculate_execution_time
    def defense_init(self):    
        bounds = [-1 ,1]

        # Get victim model
        self.origin_victim_model, self.origin_victim_model_file_name = self.train_victim()

        # Wrap victim model class
        self.victim_model = Victim_Model(
            origin_model=self.origin_victim_model,
            bounds=bounds,
            num_classes=nclasses_dict[self.dataset]
        )
    
    def evaluate(self, test_loader):
        return test(self.victim_model, test_loader)
    

    def query(self, x, soft_label=True, log_score=False):
        batch_size = self.batch_size
        self.victim_model.eval()
        n = x.size(0)
        
        # Initialize an empty tensor for storing results
        y_pred = torch.empty((n,)).to(self.device)

        # Process the data in batches
        for i in range(0, n, batch_size):
            x_batch = x[i:i+batch_size].to(self.device)
            y_batch = self.victim_model(x_batch, label=not soft_label)
            y_pred[i:i+batch_size] = y_batch

        return y_pred


    def get_log_score(self):
        whole_score = torch.cat(self.score_log_list, dim=0)
        return whole_score

    def get_query_cost(self):
        cnt = 0
        for score in self.score_log_list:
            cnt += score.size(0)

        return cnt

    def get_budget_left(self):
        budget_left = self.budget - self.get_query_cost()
        return budget_left

    def clean_log(self):
        self.score_log_list = []


    def get_layer_embeddings(self, data_loader, n_samples=1000)->torch.Tensor:
        embeddings_list = None
        cnt = 0
        
        # 创建 tqdm 对象
        pbar = tqdm(total=n_samples, desc="Processing layer embeddings")


        for x, y in data_loader:
            x = x.to(self.device)
            self.victim_model.eval()
            _, layer_embeddings_list = self.victim_model.get_layer_embeddings(x)
            if embeddings_list:
                for layer in range(len(layer_embeddings_list)):
                    embeddings_list[layer].append(layer_embeddings_list[layer])
            else:
                embeddings_list = []
                for layer in range(len(layer_embeddings_list)):
                    embeddings_list.append([layer_embeddings_list[layer]])
            
            cnt += x.size(0)
            pbar.update(x.size(0))  # 手动更新进度
            if cnt >= n_samples:
                break

        pbar.close()  # 关闭 tqdm 对象

        res_list = []
        for layer_list in embeddings_list:
            tmp = torch.cat(layer_list, dim=0)[:n_samples]
            res_list.append(tmp)
        return res_list

