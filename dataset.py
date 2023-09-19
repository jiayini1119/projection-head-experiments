import torchvision.transforms as transforms
from spuco.datasets import WILDSDatasetWrapper
from wilds import get_dataset
    
    
def get_split_dataset(named_dataset, augment: bool=True):
    # Load the full dataset, and download it if necessary
    dataset = get_dataset(dataset=named_dataset, download=True, root_dir='/home/data')

    target_resolution = (224, 224)
    scale = 256.0 / 224.0

    if augment:
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
    
    else:
        transform_train = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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

    if named_dataset == "waterbirds":
        trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="background", verbose=True)
        testset = WILDSDatasetWrapper(dataset=test_data, metadata_spurious_label="background", verbose=True)
        valset = WILDSDatasetWrapper(dataset=val_data, metadata_spurious_label="background", verbose=True)
    
    elif named_dataset == "celebA":
        trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="male", verbose=True)
        testset = WILDSDatasetWrapper(dataset=test_data, metadata_spurious_label="male", verbose=True)
        valset = WILDSDatasetWrapper(dataset=val_data, metadata_spurious_label="male", verbose=True)
    
    else:
        raise ValueError("unsupported dataset")
    
    return trainset, testset, valset