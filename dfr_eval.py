import torch 
import argparse
import wandb
import random
from spuco.datasets import WILDSDatasetWrapper
from spuco.datasets import GroupLabeledDatasetWrapper
from spuco.utils import set_seed
from wilds import get_dataset
import torchvision.transforms as transforms
import numpy as np

from models.model_factory import model_factory 
from evaluator import Evaluator_PH
from dfr import DFR_PH


from spuco.utils.random_seed import seed_randomness


def main(args):
    # wandb.init(
    #     project="DFR",
    #     config=args
    # )

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    # Load the full dataset, and download it if necessary
    dataset = get_dataset(dataset="waterbirds", download=True, root_dir="/home/data")

    target_resolution = (224, 224)
    scale = 256.0 / 224.0
    transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=transform
    )

    # Get the test set
    test_data = dataset.get_subset(
        "test",
        transform=transform
    )

    # Get the val set
    val_data = dataset.get_subset(
        "val",
        transform=transform
    )

    trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="background", verbose=True)
    testset = WILDSDatasetWrapper(dataset=test_data, metadata_spurious_label="background", verbose=True)
    valset = WILDSDatasetWrapper(dataset=val_data, metadata_spurious_label="background", verbose=True)

    if args.no_ph:
        args.use_ph = False


    # load a trained model from checkpoint
    if args.no_ph:
        model = model_factory("resnet50", trainset[0][0].shape, 2).to(device)
    else:
        model = model_factory("resnet50", trainset[0][0].shape, 2, hidden_dim=2048).to(device)
    ckpt_path = args.model_path
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)

    # print(model)

    if args.subset_size is None:
        group_labeled_set = GroupLabeledDatasetWrapper(dataset=valset, group_partition=valset.group_partition)
    else:
        subset_indices = random.sample(range(len(valset)), args.subset_size)
        
        group_labeled_set = GroupLabeledDatasetWrapper(dataset=valset, group_partition=valset.group_partition, subset_indices=subset_indices)

    dfr = DFR_PH(
        group_labeled_set=group_labeled_set,
        model=model,
        data_for_scaler=trainset,
        device=device,
        verbose=True,
        use_ph=args.use_ph,
    )

    dfr.train()

    evaluator = Evaluator_PH(
        testset=testset,
        group_partition=testset.group_partition,
        group_weights=trainset.group_weights,
        batch_size=args.test_batch_size,
        model=model,
        sklearn_linear_model=dfr.linear_model,
        device=device,
        verbose=False,
        use_ph=args.use_ph,
        )

    evaluator.evaluate()

    print(evaluator.worst_group_accuracy)
    print(evaluator.average_accuracy)

    # wandb.log({"worst group accuracy": evaluator.worst_group_accuracy})
    # wandb.log({"average accuracy": evaluator.average_accuracy})

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='erm pretrain')
    parser.add_argument('--device', type=int, default=7, help="GPU number")
    parser.add_argument("--test-batch-size", type=int, default=64, help='Testing batch size')
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")
    parser.add_argument("--use-ph", action='store_true', help='Whether to use post projection head representationin DFR')
    parser.add_argument("--test-freq", type=int, default=20, help='Test frequency')
    parser.add_argument("--model-path", type=str, default=None, help='Path for the pretrained model')
    parser.add_argument("--subset-size", type=int, default=None, help='size of subset of the labeled dataset')
    parser.add_argument("--no-ph", action='store_true', help='Whether the model has no projection head')

    args = parser.parse_args()

    main(args)

