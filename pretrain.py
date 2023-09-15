"""
Pretrain with projection head
"""
import torch 
import numpy as np
import argparse
from datetime import datetime
from spuco.utils import set_seed
from wilds import get_dataset
import torchvision.transforms as transforms
from spuco.datasets import WILDSDatasetWrapper

from models.model_factory import *
from evaluator import Evaluator_PH
from trainer.erm import ERM 
from trainer.supervised_cl import SCL
from torch.optim import SGD
from loss.supcon_loss import SupConLoss

def main(args):

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")


    DT_STRING = "".join(str(datetime.now()).split())

    # Load the full dataset, and download it if necessary
    dataset = get_dataset(dataset="waterbirds", download=True, root_dir='/home/data')

    target_resolution = (224, 224)
    transform_train = transforms.Compose([
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    scale = 256.0 / 224.0
    transform_test = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=transform_train
    )

    # Get the test set
    test_data = dataset.get_subset(
        "test",
        transform=transform_test
    )

    # Get the val set
    val_data = dataset.get_subset(
        "val",
        transform=transform_test
    )

    trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="background", verbose=True)
    testset = WILDSDatasetWrapper(dataset=test_data, metadata_spurious_label="background", verbose=True)
    valset = WILDSDatasetWrapper(dataset=val_data, metadata_spurious_label="background", verbose=True)


    if not args.without_ph:
        # model with projection head
        if args.random_init:
            model = model_factory("resnet50", trainset[0][0].shape, 2, pretrained=False, hidden_dim=2048).to(device)
        else:
            model = model_factory("resnet50", trainset[0][0].shape, 2, pretrained=True, hidden_dim=2048).to(device)
    else:
        if args.random_init:
            model = model_factory("resnet50", trainset[0][0].shape, 2, pretrained=False).to(device)
        else:
            model = model_factory("resnet50", trainset[0][0].shape, 2, pretrained=True).to(device)
    
    print(model)

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
            criterion=SupConLoss(),
            val_evaluator=val_evaluator,
            batch_size=args.train_batch_size,
            optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9),
            device=device,
            verbose=not args.without_ph
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

    torch.save(model.state_dict(), f'pretrained_model_{args.pretrain_method}_{DT_STRING}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='erm pretrain')
    parser.add_argument('--device', type=int, default=7, help="GPU number")
    parser.add_argument('--lr', type=float, default=1e-03, help='learning rate') 
    parser.add_argument('--weight-decay', type=float, default=1e-03, help='learning rate') 
    parser.add_argument("--train-batch-size", type=int, default=32, help='Training batch size')
    parser.add_argument("--test-batch-size", type=int, default=64, help='Testing batch size')
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")
    parser.add_argument('--num_epochs', type=int, default=100, help="number of epochs to train")
    parser.add_argument("--test-freq", type=int, default=20, help='Test frequency')
    parser.add_argument("--pretrain-method", type=str, default="ERM", choices=['ERM', 'SCL'], help='pretrain method')
    parser.add_argument("--without-ph", action='store_true', help='Whether to not use projection head')
    parser.add_argument("--random-init", action='store_true', help='Whether to use random initialization')


    args = parser.parse_args()

    main(args)