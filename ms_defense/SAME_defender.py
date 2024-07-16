import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import math
from einops import rearrange

from ml_models import get_model, get_ood_detector, get_configs, VGGAutoEncoder, ResNetAutoEncoder, MAE_ViT
from ml_datasets import get_dataloaders, ds_choices, nclasses_dict, nch_dict, get_autoencoder_dataloaders, xdim_dict
from ml_common import train, get_device, train_with_ood, calculate_execution_time, train_with_generator, print_execution_time
from utils import D3Model, Victim_Model
from tqdm import tqdm

from ms_defense import Base_Defender

import os

class SAME_Defender(Base_Defender):

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
        defense_name='SAME',
        budget=0,
        alpha=0.1,
        emb_dim=192,
        mae_epochs=500
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
        self.mae_epochs = mae_epochs

        self.origin_victim_model = None
        self.origin_victim_model_file_name = None
        self.device = get_device()
        self.budget = budget
        self.alpha=alpha
        self.emb_dim=emb_dim
    
    @print_execution_time
    def train_victim(self):
        self.victim_save_path = f"./exp/{self.dataset}/{self.exp_id}/{self.defense_name}/victim/"
        model_file_name = f"{self.victim_save_path}/{self.model_name}-{self.epochs}.pt"
        if not os.path.exists(self.victim_save_path):
                os.makedirs(self.victim_save_path)
        
        model = get_model(self.model_name, self.dataset, pretrained=False)
        model = model.to(self.device)

        if os.path.exists(model_file_name):
            print("Victim model already exists. Load from file:",model_file_name)
            model.load_state_dict(torch.load(model_file_name))
        else:
            dataloader_train, dataloader_test = get_dataloaders(self.dataset, self.batch_size, augment=self.augment)

            criterion = nn.CrossEntropyLoss()
            
            print(f"\n==Training Victim Model==")

            opt = SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            sch = CosineAnnealingLR(opt, self.epochs, last_epoch=-1)

            train(
                model=model,
                train_loader=dataloader_train,
                test_loader=dataloader_test,
                criterion=criterion,
                opt=opt,
                epochs=self.epochs,
                sch=sch,
            )
            
            torch.save(model.state_dict(), model_file_name)

        return model, model_file_name
    
    @calculate_execution_time
    def defense_init(self):
        bounds = [-1, 1]
    
        # Get victim model
        self.origin_victim_model, self.origin_victim_model_file_name = self.train_victim()
        # Construct D3Model
        self.victim_model = Victim_Model(
            origin_model=self.origin_victim_model,
            bounds=bounds,
            num_classes=nclasses_dict[self.dataset]
        )
        
        # Train Auto-encoder-based OOD classifier
        self.autoencoder = self.train_autoencoder(total_epochs=self.mae_epochs)

        # Train shadow classifier
        self.shadow_classifier, self.shadow_classifier_file_name = self.train_shadow_classifier(epochs=80)

    
    def train_autoencoder(self, total_epochs=50, warmup_epochs=50, lr=0.05, loss="mse", mask_ratio=0.75):
        """
        This function is modified from: https://github.com/IcarusWizard/MAE/blob/main/mae_pretrain.py
        """
        model_log_list = [1,2,3,4,5,10,20,30,40,50,100,200,300,400,500]


        # Define the architecture of auto-encoder
        input_dim = nch_dict[self.dataset]
        image_size = xdim_dict[self.dataset]
        mae_model = self.get_masked_autoencoder(input_dim=input_dim,image_size=image_size,emb_dim=self.emb_dim)
        mae_model = mae_model.to(self.device)

        self.autoencoder_save_path = f"./exp/{self.dataset}/{self.exp_id}/{self.defense_name}/autoencoder/"

        if not os.path.exists(self.autoencoder_save_path):
            os.makedirs(self.autoencoder_save_path)

        self.autoencoder_file_name = self.autoencoder_save_path + f"mae-{self.emb_dim}-{total_epochs}.pt"
        if os.path.exists(self.autoencoder_file_name):
            print("Autoencoder model already exists. Load from file:",self.autoencoder_file_name)
            mae_model.load_state_dict(torch.load(self.autoencoder_file_name))

            return mae_model

        opt = torch.optim.AdamW(mae_model.parameters(), lr=0.01, betas=(0.9, 0.95), weight_decay=0.05)
        lr_func = lambda epoch: min((epoch + 1) / (warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / total_epochs * math.pi) + 1))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_func, verbose=True)

        # Define dataloader
        dataloader_norm, dataloader_test = get_autoencoder_dataloaders(self.dataset, self.batch_size, augment=False)

        # Sample some images to be used for visualization in each epoch
        fixed_sample_images, _ = next(iter(dataloader_test))
        fixed_sample_images = fixed_sample_images.to(self.device)

        # Start to train
        self.image_save_path = f"./exp/{self.dataset}/{self.exp_id}/{self.defense_name}/images/"


        step_count = 0
        opt.zero_grad()
        for epoch in range(total_epochs):
            mae_model.train()
            losses = []
            for img, label in tqdm(iter(dataloader_norm)):
                step_count += 1
                img = img.to(self.device)
                predicted_img, mask = mae_model(img)
                # loss = torch.mean((predicted_img - img) ** 2 * mask) / mask_ratio
                loss = torch.mean((predicted_img - img) ** 2) / mask_ratio
                loss.backward()
                opt.step()
                opt.zero_grad()
                losses.append(loss.item())
            lr_scheduler.step()
            avg_loss = sum(losses) / len(losses)
            print(f"[epoch:{epoch}/{total_epochs}] mae_loss: {avg_loss:.6f}")

            if epoch in model_log_list:
                log_name = f"mae-{self.emb_dim}-{epoch}"
                with torch.no_grad():
                    reconstructed_images, mask = mae_model(fixed_sample_images)
                    visualize_mask_images(fixed_sample_images, reconstructed_images, mask, epoch, self.image_save_path, log_name)
                
                torch.save(mae_model.state_dict(), self.autoencoder_save_path + log_name+f".pt")

        # Test and Save

        torch.save(mae_model.state_dict(), self.autoencoder_file_name)

        return mae_model
    
    def train_shadow_classifier(self, epochs:int=10):
        self.model_save_path = f"./exp/{self.dataset}/{self.exp_id}/{self.defense_name}/shadow/"
        model_file_name = f"{self.model_save_path}/{self.model_name}-{epochs}.pt"
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        
        shadow_classifier = get_model(self.model_name, self.dataset, pretrained=False)
        shadow_classifier = shadow_classifier.to(self.device)

        if os.path.exists(model_file_name):
            print("Shadow classifier already exists. Load from file:",model_file_name)
            shadow_classifier.load_state_dict(torch.load(model_file_name))

            return shadow_classifier, model_file_name

        dataloader_train, dataloader_test = get_dataloaders(self.dataset, self.batch_size, augment=self.augment)
        

        criterion = nn.CrossEntropyLoss()

        print(f"\n==Training Shadow Classifier==")

        opt = SGD(shadow_classifier.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        sch = CosineAnnealingLR(opt, epochs, last_epoch=-1)

        train_with_generator(
            model=shadow_classifier,
            generator=self.autoencoder,
            train_loader=dataloader_train,
            test_loader=dataloader_test,
            criterion=criterion,
            opt=opt,
            epochs=epochs,
            sch=sch,
        )
        
        torch.save(shadow_classifier.state_dict(), model_file_name)

        return shadow_classifier, model_file_name


    def get_autoencoder(self, arch="vgg11", input_dim=3):
        config, bottleneck = get_configs(arch=arch)
        if arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
            model = VGGAutoEncoder(config, input_dim=input_dim)
        elif arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
            model = ResNetAutoEncoder(config, bottleneck)
        else:
            raise ValueError

        return model

    def get_masked_autoencoder(self, input_dim=3,image_size=32, emb_dim=192):
        """
            Get the masked autoencoder.
        """

        model = MAE_ViT(mask_ratio=0.75,n_channel=input_dim,image_size=image_size, emb_dim=emb_dim)

        return model
                    
    def _judge_ood(self, x, criterion=nn.MSELoss(), tau=0.5, dim_mean=False):
        raise NotImplementedError

    def _judge_ood_score(self, x, criterion_1=nn.MSELoss(reduction='none'),criterion_2=nn.KLDivLoss(reduction='none'), alpha=0.1, dim_mean=False):
        with torch.no_grad():
            x = x.to(self.device)
            reconstruct_x, mask = self.autoencoder(x, mask_ratio=0)
            score_1 = criterion_1(reconstruct_x, x).detach().cpu()


            shadow_pred = self.shadow_classifier(reconstruct_x.detach())
            shadow_pred = F.log_softmax(shadow_pred, dim=1)
            pred = self.victim_model(x)
            pred = F.softmax(pred, dim=1)

            score_2 = criterion_2(shadow_pred, pred).detach().cpu()
            if dim_mean:
                dim_size = len(score_1.size())
                dim_list = [i+1 for i in range(dim_size-1)]
                score_1 = score_1.mean(dim=dim_list)
                dim_size = len(score_2.size())
                dim_list = [i+1 for i in range(dim_size-1)]
                score_2 = score_2.mean(dim=dim_list)
            
            assert len(score_1.size())==1, f"The len {len(score_1.size())} is not equal to 1"
            assert len(score_2.size())==1, f"The len {len(score_2.size())} is not equal to 1"
            score = alpha * score_1 + (1-alpha) * score_2

        return score

    def get_ood_score(self, data_loader, criterion='mse') -> torch.Tensor:
        res_list = []
        if criterion == 'mse':
            criterion = nn.MSELoss(reduction='none')
            dim_mean = True
        else:
            raise NotImplementedError

        for (x,y) in tqdm(data_loader, ncols=80, leave=False, desc="OOD score calculate..."):
            score = self._judge_ood_score(x, criterion=criterion, dim_mean=dim_mean)
            res_list.append(score)
        
        whole_score = torch.cat(res_list, dim=0)

        return whole_score

    
    def ood_test(self, data_loader, criterion='mse', tau=0.1) -> int:
        raise NotImplementedError


    def query(self, x, soft_label=True, log_score=False):
        batch_size = self.batch_size
        n = x.size(0)

        y_pred_list = []

        for i in range(0, n, batch_size):
            x_batch = x[i:i+batch_size].to(self.device)
            y_batch = self.query_batch_step(x_batch, soft_label=soft_label,log_score=log_score)
            y_pred_list.append(y_batch)

        y = torch.cat(y_pred_list,dim=0)
    
        return y



    def query_batch_step(self, x, soft_label=True, log_score=False):
        if log_score:
            assert self.get_budget_left() >= x.size(0), f"Budget left {self.get_budget_left()} should >= batch_size {x.size(0)}"
            with torch.no_grad():
                score = self._judge_ood_score(x, dim_mean=True, alpha=self.alpha)
                self.score_log_list.append(score)

        x = x.to(self.device)
        self.victim_model.eval()
        y = self.victim_model(x, label=not soft_label)
        
        return y



import torchvision
import os

def visualize_images(original_images, reconstructed_images, epoch, directory, num=16):
    # Make sure the directory exists
    os.makedirs(directory, exist_ok=True)

    if original_images.size(0) > num:
        original_images = original_images[:num]
        reconstructed_images = reconstructed_images[:num]

    # We'll use torchvision's save_image function, which expects images in the range [0, 1]
    # If your images are in the range [-1, 1], you can rescale them to [0, 1] like this:
    if epoch<= 1:
        original_images = (original_images + 1) / 2
        torchvision.utils.save_image(original_images, os.path.join(directory, f'original_epoch{epoch}.png'), nrow=4)
    
    reconstructed_images = (reconstructed_images + 1) / 2    
    torchvision.utils.save_image(reconstructed_images, os.path.join(directory, f'reconstructed_epoch{epoch}.png'), nrow=4)


def visualize_mask_images(original_images, reconstructed_images, mask, epoch, directory, log_name,num=16):
    # Make sure the directory exists
    os.makedirs(directory, exist_ok=True)

    if original_images.size(0) > num:
        original_images = original_images[:num]
        reconstructed_images = reconstructed_images[:num]
        mask = mask[:num]
    # reconstructed_images = reconstructed_images * mask + original_images * (1-mask)
    img = torch.cat([original_images * (1 - mask), reconstructed_images ,original_images], dim=0)
    img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
    img = (img + 1) / 2

    torchvision.utils.save_image(img, os.path.join(directory, f'image-{log_name}.png'))