"""Extract embeddings from an ImageNet-like dataset split and save to disk.
"""

import torch
import torchvision
import numpy as np
import tqdm
import argparse
import os
import imagenet_datasets
from models.model_factory import model_factory 


parser = argparse.ArgumentParser(
    description="Extract embeddings from an ImageNet-like dataset split and "
              "save to disk.")
parser.add_argument(
    "--dataset", type=str, default="bg_challenge",
    help="Dataset variation [imagenet | imagenet-a | imagenet-r | imagenet-c "
         "| bg_challenge]")
parser.add_argument(
    "--dataset_dir", type=str, default="/datasets/imagenet-stylized/",
    help="ImageNet dataset directory")
parser.add_argument(
    "--split", type=str, default="train",
    help="ImageNet dataset directory")
parser.add_argument(
    "--batch_size", type=int, default=500,
    help="Batch size")
parser.add_argument("--use_prev_block", action='store_true', 
    help='Whether to use representation before previous blocks')
parser.add_argument('--device', type=int, default=7, help="GPU number")
parser.add_argument('--pretrain_method', type=str, default="standardSL", choices=['standardSL', 'simclr'], help='which pretrained model to use')
parser.add_argument("--without_ph", action='store_true', help='Whether to not use projection head')
parser.add_argument("--use-ph", action='store_true', help='Whether to use post projection head representationin DFR')

args = parser.parse_args()

device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

def get_embed(m, x, use_prev_block: bool=False, without_ph: bool=True, use_ph: bool=False):
    if not without_ph and use_prev_block:
        raise ValueError("Cannot use previous block and projection head at the same time")
    
    if without_ph:
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)

        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        if not use_prev_block:
            x = m.layer4(x)
            x = m.avgpool(x)
            x = torch.flatten(x, 1)
        else:
            x = m.layer4[0](x)
            x = m.layer4[1](x)
            bottleneck = m.layer4[2]
            x = bottleneck.conv1(x)
            x = bottleneck.bn1(x)
            x = bottleneck.relu(x)
            x = bottleneck.conv2(x)
            x = bottleneck.bn2(x)
            x = bottleneck.relu(x)
            x = m.avgpool(x)
            x = torch.flatten(x, 1)
        return x

    else:
        x = m.get_representation(x, use_ph=use_ph)
        return x

resize_size, crop_size = 256, 224


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(resize_size),
    torchvision.transforms.CenterCrop(crop_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
])

ds, loader = imagenet_datasets.get_imagenet_like(
    name=args.dataset,
    datapath=args.dataset_dir,
    split=args.split,
    transform=transform,
    batch_size=args.batch_size,
    shuffle=False
)

if args.without_ph:
    if args.pretrain_method == "standardSL":
        model = torchvision.models.resnet50(pretrained=True).to(device)

        # # load the finetuned model
        # model = torchvision.models.resnet50(pretrained=False).to(device)
        # checkpoint = torch.load("/home/jennyni/projection-head-experiments/original_pretrained_model.pt", map_location=device)
        # state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        # model.load_state_dict(state_dict)



    else:
        model = torchvision.models.resnet50(pretrained=False).to(device)
        # TODO: remove projection head
        state_dict = torch.load("path-to-simclr-model", map_location=device)
        model.load_state_dict(state_dict)
    
else:
    model = model_factory("resnet50", ds[0][0].shape, 1000, hidden_dim=2048).to(device)
    checkpoint = torch.load("/home/jennyni/projection-head-experiments/imagenet_pretrained_model_2023-09-2020:05:35.080839.pt", map_location=device)
    # checkpoint = torch.load("/home/jennyni/projection-head-experiments/imagenet_pretrained_model_2023-09-2118:46:39.757465.pt", map_location=device)  
    # checkpoint = torch.load("/home/jennyni/projection-head-experiments/imagenet_pretrained_model_2023-09-2118:46:44.362153.pt", map_location=device)  
    # checkpoint = torch.load("/home/jennyni/projection-head-experiments/imagenet_pretrained_model_2023-09-2205:40:56.865258.pt", map_location=device)  
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)


model.eval()

all_embeddings = []
all_y = []

not_printed = True
for x, y in tqdm.tqdm(loader):
    with torch.no_grad():
        embed = get_embed(model, x.to(device), use_prev_block=args.use_prev_block, without_ph=args.without_ph, use_ph=args.use_ph).detach().cpu().numpy() 
        all_embeddings.append(embed)
        all_y.append(y.detach().cpu().numpy())
        if not_printed:
            print("Embedding shape:", embed.shape)
            not_printed = False

all_embeddings = np.vstack(all_embeddings)
all_y = np.concatenate(all_y)



# np.savez(os.path.join(
#         args.dataset_dir,
#         f"new_ph_{args.dataset}_{args.use_ph}_{args.split}_{args.pretrain_method}_embeddings.npz"),
#     embeddings=all_embeddings,
#     labels=all_y)


np.savez(os.path.join(
        args.dataset_dir,
        f"check_{args.dataset}_{args.use_prev_block}_{args.split}_{args.pretrain_method}_embeddings.npz"),
    embeddings=all_embeddings,
    labels=all_y)
