import argparse

import torch, os
from ms_defense import Base_Defender, SAME_Defender
from ms_attacks import Normal_User, JBDA_Attacker, MS_Attacker, DFME_Attacker, Knockoff_Attacker
from ml_models import model_choices
from ml_datasets import get_dataloaders, ds_choices


from sklearn.metrics import precision_recall_curve, auc, roc_curve
from scipy.interpolate import interp1d

def main():

    # Load parameters
    parser = argparse.ArgumentParser(description="train victim model")

    parser.add_argument("--exp_id", type=str, default="main", help="The name of this experiments")
    parser.add_argument("--model", type=str, default='res18', help="model architecture")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument("--dataset_ood", type=str, default="imagenet_tiny", help="OOD dataset used by the defender")
    parser.add_argument("--proxyset", type=str, default="cifar100", help="Proxyset that used by the attacker.")
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size of the dataloader")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate of the training")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs of model")
    parser.add_argument("--budget", type=int, default=5000, help="Query budget of each attacker")
    parser.add_argument("--attacker", type=str, default='knockoff', help="Attack strategy", choices=['knockoff', 'dfme', 'jbda'])
    parser.add_argument("--defenders", type=str, default='SAME', help="Defender name list")
    parser.add_argument("--alpha", type=float, default=0.99, help="Weight of score_1 (reconstruction mse loss). Total_score=alpha*score_1 + (1-alpha) * score_2.")
    parser.add_argument("--mae_epochs", type=int, default=500, help="Training epochs of masked autoencoder.")
    
    args = parser.parse_args()

    # Initialize datasets
    train_loader, query_loader = get_dataloaders(
        args.dataset, args.batch_size, augment=True
    )
    ood_loader, _ = get_dataloaders(
        args.dataset_ood, args.batch_size, augment=True
    )
    proxy_loader, _ = get_dataloaders(
        args.proxyset, args.batch_size, augment=False
    )

    # Initialize Normal User
    normal_user = Normal_User(
        exp_id=args.exp_id,
        dataset=args.dataset,
        batch_size=args.batch_size
    )

    # Initialize Attacker - Knockoff
    if args.attacker == 'knockoff':
        attacker = Knockoff_Attacker(
            exp_id=args.exp_id,
            model=args.model,
            dataset=args.dataset,
            dataset_proxy=args.proxyset,
            batch_size=args.batch_size,
            lr=args.lr
        )
    elif args.attacker == 'jbda':
        attacker = JBDA_Attacker(
            exp_id=args.exp_id,
            model=args.model,
            dataset=args.dataset,
            dataset_proxy=args.proxyset,
            batch_size=args.batch_size,
            lr=args.lr
        )
    elif args.attacker == 'dfme':
        attacker = DFME_Attacker(
            exp_id=args.exp_id,
            model=args.model,
            dataset=args.dataset,
            batch_size=args.batch_size,
            lr=args.lr
        )
    else:
        raise ValueError

    if 'SAME' in args.defenders:
        defender = MaRD_Defender(
            exp_id=args.exp_id,
            model=args.model,
            dataset=args.dataset,
            dataset_ood=args.dataset_ood,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            augment=True,
            budget=args.budget,
            alpha=args.alpha,
            mae_epochs=args.mae_epochs
        )
        _, cost_time = defender.defense_init()
        clean_acc = defender.evaluate(query_loader)
        print(f"Clean_acc of {defender.defense_name} is {clean_acc*100:.2f}%, cost time: {cost_time:.1f} s")

        attacker.attack_init()
        attack_test(defender, attacker, normal_user, args.budget, args.exp_id)
        del defender



def attack_test(defender:Base_Defender, attacker:MS_Attacker, normal_user:MS_Attacker, budget:int, exp_id:str='exp'):
    defender.clean_log()
    
    attacker.steal(
        victim=defender,
        budget=budget,
        visual_log=True,
    )

    ood_score = defender.get_log_score()

    defender.clean_log()

    normal_user.query(
        victim=defender,
        budget=budget
    )

    clean_score = defender.get_log_score()
    substitute_acc = attacker.get_accuracy()

    # Calculate FPR95, FPR90, AUROC, AUPR
    fpr, tpr, thresholds, fpr95, fpr90, auroc, aupr = get_roc(clean_score, ood_score)
    print(f"[{exp_id}][Attacker: {attacker.attack_name}:Budget={budget}:ACC={substitute_acc*100:.2f}][Defender:{defender.defense_name}]AUROC: {auroc*100:.2f}, AUPR: {aupr*100:.2f}, FPR95: {fpr95*100:.2f}, FPR90: {fpr90*100:.2f}")

def get_roc(clean_score, ood_score):
    clean_label = torch.zeros(clean_score.size())
    adv_label = torch.ones(ood_score.size())

    score_tensor = torch.cat((clean_score, ood_score), dim=0)
    label_tensor = torch.cat((clean_label, adv_label), dim=0)

    # Calculate ROC
    fpr, tpr, thresholds = roc_curve(label_tensor, score_tensor)

    # Calculate FPR at 95% TPR
    f = interp1d(tpr, fpr)
    fpr95 = f(0.95)
    fpr90 = f(0.90)

    # Calculate AUPR
    precision, recall, _ = precision_recall_curve(label_tensor, score_tensor)
    auroc = auc(fpr, tpr)
    aupr = auc(recall, precision)

    return fpr, tpr, thresholds, fpr95, fpr90, auroc, aupr

if __name__ == "__main__":
    main()