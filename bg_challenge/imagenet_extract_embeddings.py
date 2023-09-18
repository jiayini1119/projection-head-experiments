"""Extract embeddings from an ImageNet-like dataset split and save to disk.
"""

import torch
import torchvision
import numpy as np
import tqdm
import argparse
import os
import imagenet_datasets

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




args = parser.parse_args()

device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

if args.pretrain_method == "standardSL":
    model = torchvision.models.resnet50(pretrained=True).to(device)
else:
    model = torchvision.models.resnet50(pretrained=False).to(device)
    # TODO: remove projection head
    state_dict = torch.load("path-to-simclr-model")
    model.load_state_dict(state_dict)



def get_embed(m, x, use_prev_block: bool=False):
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
        x = m.avgpool(x)
        x = torch.flatten(x, 1)
    return x
resize_size, crop_size = 256, 224


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(resize_size),
    torchvision.transforms.CenterCrop(crop_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
])


model.eval()

ds, loader = imagenet_datasets.get_imagenet_like(
    name=args.dataset,
    datapath=args.dataset_dir,
    split=args.split,
    transform=transform,
    batch_size=args.batch_size,
    shuffle=False
)

all_embeddings = []
all_y = []

not_printed = True
for x, y in tqdm.tqdm(loader):
    with torch.no_grad():
        embed = get_embed(model, x.to(device), use_prev_block=args.use_prev_block).detach().cpu().numpy() 
        all_embeddings.append(embed)
        all_y.append(y.detach().cpu().numpy())
        if not_printed:
            print("Embedding shape:", embed.shape)
            not_printed = False

all_embeddings = np.vstack(all_embeddings)
all_y = np.concatenate(all_y)



np.savez(os.path.join(
        args.dataset_dir,
        f"{args.dataset}_{args.use_prev_block}_{args.split}_{args.pretrain_method}_embeddings.npz"),
    embeddings=all_embeddings,
    labels=all_y)
