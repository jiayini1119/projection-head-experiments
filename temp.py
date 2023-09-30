import numpy as np

# Load the .npz file
data = np.load("/home/jennyni/datasets/imagenet-r/new_ph_imagenet-r_False_val_standardSL_embeddings.npz")

array_keys = data.files
print("Arrays in the .npz file:")
for key in array_keys:
    print(key)

for key in array_keys:
    array = data[key]
    print(f"\nContents of array '{key}':")
    print(array)

data.close()