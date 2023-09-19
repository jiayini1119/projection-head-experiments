import torch 
import argparse
from spuco.utils import set_seed
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.model_factory import *
from torch.optim import SGD

def data_loader(root, batch_size=256):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=5,
        pin_memory=True,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=5,
        pin_memory=True
    )
    return train_loader, val_loader, train_dataset


def main(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    train_loader, val_loader, train_dataset = data_loader("/home/jennyni/datasets/imagenet", args.batch_size)

    model = model_factory("resnet50", train_dataset[0][0].shape, 1000, pretrained=True, hidden_dim=2048).to(device)

    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        print("Train loss: {:.3f}".format(loss))

        if (args.test_freq > 0) and ((epoch + 1) % args.test_freq == 0):
            model.eval()
            correct_num = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    predicted = torch.argmax(output, dim=1)
                    total += target.size(0)
                    correct_num += (predicted == target).sum().item()

            val_accuracy = correct_num / total

            print("Epoch: {:d}, Validation accuracy: {:.3f}".format(epoch + 1, val_accuracy))


    torch.save(model.state_dict(), "imagenet_pretrained_model.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='erm pretrain')
    parser.add_argument('--device', type=int, default=7, help="GPU number")
    parser.add_argument('--batch-size', type=int, default=256, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-03, help='learning rate') 
    parser.add_argument('--weight-decay', type=float, default=1e-03, help='learning rate') 
    parser.add_argument('--seed', type=int, default=0, help="Seed for randomness")
    parser.add_argument('--num-epochs', type=int, default=100, help="number of epochs to train")
    parser.add_argument("--test-freq", type=int, default=20, help='Test frequency')


    args = parser.parse_args()

    main(args)