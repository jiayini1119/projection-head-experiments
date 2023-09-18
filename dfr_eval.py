import torch 
import argparse
import wandb
import random
from spuco.datasets import GroupLabeledDatasetWrapper
from spuco.utils import set_seed
import numpy as np

from models.model_factory import model_factory 
from evaluator import Evaluator_PH
from dfr import DFR_PH

from dataset import get_split_dataset
from spuco.utils.random_seed import seed_randomness


def main(args):
    # wandb.init(
    #     project="DFR",
    #     config=args
    # )

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    trainset, testset, valset = get_split_dataset(named_dataset=args.dataset)

    print(len(valset))

    if args.no_ph:
        args.use_ph = False

    # load a trained model from checkpoint
    if args.no_ph:
        model = model_factory("resnet50", trainset[0][0].shape, 2).to(device)
    else:
        model = model_factory("resnet50", trainset[0][0].shape, 2, hidden_dim=2048, mult_layer=args.mult_layer).to(device)
    ckpt_path = args.model_path
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)

    print(model)

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

    print(evaluator.accuracies)
    # wandb.log({"worst group accuracy": evaluator.worst_group_accuracy})
    # wandb.log({"average accuracy": evaluator.average_accuracy})

    return evaluator.worst_group_accuracy[1], evaluator.average_accuracy

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='erm pretrain')
    parser.add_argument('--device', type=int, default=7, help="GPU number")
    parser.add_argument('--dataset', type=str, default="waterbirds", choices=['waterbirds', 'celebA'], help='dataset to use')
    parser.add_argument("--test-batch-size", type=int, default=64, help='Testing batch size')
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")
    parser.add_argument("--use-ph", action='store_true', help='Whether to use post projection head representationin DFR')
    parser.add_argument("--mult-layer", action='store_true', help='Whether to use projection head with two hidden layers')
    parser.add_argument("--test-freq", type=int, default=20, help='Test frequency')
    parser.add_argument("--model-path", type=str, default=None, help='Path for the pretrained model')
    parser.add_argument("--subset-size", type=int, default=None, help='size of subset of the labeled dataset')
    parser.add_argument("--no-ph", action='store_true', help='Whether the model has no projection head')

    args = parser.parse_args()

    main(args)

    # # Try multiple seeds

    # seeds = [0, 10, 20, 30, 40]  
    # worst_group_accuracies = []
    # average_accuracies = []

    # for seed in seeds:
    #     args = parser.parse_args()
    #     args.seed = seed  
    #     worst_group_accuracy, average_accuracy = main(args)
        
    #     worst_group_accuracies.append(worst_group_accuracy)
    #     average_accuracies.append(average_accuracy)

    # avg_worst_group_accuracy = np.mean(worst_group_accuracies)
    # std_worst_group_accuracy = np.std(worst_group_accuracies)
    
    # avg_average_accuracy = np.mean(average_accuracies)
    # std_average_accuracy = np.std(average_accuracies)

    # print(f"Worst Group Accuracy: {avg_worst_group_accuracy * 100:.1f}±{std_worst_group_accuracy * 100:.1f}")
    # print(f"Average Accuracy: {avg_average_accuracy * 100:.1f}±{std_average_accuracy * 100:.1f}")



