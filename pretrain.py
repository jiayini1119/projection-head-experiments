"""
Pretrain with projection head
"""
import torch 
import numpy as np
import argparse
from datetime import datetime
from spuco.utils import set_seed

from models.model_factory import *
from evaluator import Evaluator_PH
from trainer.erm import ERM 
from trainer.supervised_cl import SCL
from torch.optim import SGD
from loss.supcon_loss import SupConLoss
from dataset import get_split_dataset

def main(args):

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    DT_STRING = "".join(str(datetime.now()).split())
    trainset, testset, valset = get_split_dataset(named_dataset=args.dataset)

    if not args.without_ph:
        # model with projection head
        model = model_factory("resnet50", trainset[0][0].shape, 2, kappa=args.kappa, pretrained=not args.random_init, hidden_dim=2048, mult_layer=args.mult_layer, identity_init=args.identity_init).to(device)
    else:
        model = model_factory("resnet50", trainset[0][0].shape, 2, pretrained=not args.random_init).to(device)
    
    val_evaluator = Evaluator_PH(
        testset=valset,
        group_partition=valset.group_partition,
        group_weights=trainset.group_weights,
        batch_size=args.test_batch_size,
        model=model,
        device=device,
        verbose=False,
        use_ph=not args.without_ph
    )

    val_evaluator.evaluate()
    print(val_evaluator.worst_group_accuracy)

    if args.pretrain_method == "ERM":
        base_trainer = ERM(
            model=model,
            num_epochs=args.num_epochs,
            trainset=trainset,
            val_evaluator=val_evaluator,
            batch_size=args.train_batch_size,
            optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9),
            device=device,
            verbose=True
        )
    else:
        base_trainer = SCL(
            model=model,
            num_epochs=args.num_epochs,
            trainset=trainset,
            criterion=SupConLoss(temperature=args.temperature),
            val_evaluator=val_evaluator,
            batch_size=args.train_batch_size,
            optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9),
            device=device,
            verbose=True,
            use_ph=not args.without_ph,
        )


    base_trainer.train()

    evaluator = Evaluator_PH(
        testset=testset,
        group_partition=testset.group_partition,
        group_weights=trainset.group_weights,
        batch_size=args.test_batch_size,
        model=base_trainer.best_model,
        device=device,
        verbose=False,
        use_ph=not args.without_ph
    )
    evaluator.evaluate()

    print(evaluator.worst_group_accuracy)

    torch.save(model.state_dict(), f'finalized_ori_{args.dataset}_{args.seed}_{args.pretrain_method}_{args.kappa}_{DT_STRING}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='erm pretrain')
    parser.add_argument('--dataset', type=str, default="waterbirds", choices=['waterbirds', 'celebA'], help='dataset to use')
    parser.add_argument('--device', type=int, default=7, help="GPU number")
    parser.add_argument('--lr', type=float, default=1e-03, help='learning rate') 
    parser.add_argument('--weight-decay', type=float, default=1e-03, help='learning rate') 
    parser.add_argument("--train-batch-size", type=int, default=32, help='Training batch size')
    parser.add_argument("--test-batch-size", type=int, default=64, help='Testing batch size')
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")
    parser.add_argument('--num-epochs', type=int, default=100, help="number of epochs to train")
    parser.add_argument("--test-freq", type=int, default=20, help='Test frequency')
    parser.add_argument("--pretrain-method", type=str, default="ERM", choices=['ERM', 'SCL'], help='pretrain method')
    parser.add_argument("--without-ph", action='store_true', help='Whether to not use projection head')
    parser.add_argument("--random-init", action='store_true', help='Whether to use random initialization')
    parser.add_argument("--identity-init", action='store_true', help='Whether to initialize projection head as identity')

    parser.add_argument("--mult-layer", action='store_true', help='Whether to use projection head with two hidden layers')
    parser.add_argument("--temperature", type=float, default=0.5, help='temperature for supervised contrastive loss')

    parser.add_argument("--kappa", type=float, default=1.05, help='kappa')



    args = parser.parse_args()

    main(args)