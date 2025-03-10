import os

dataset_path = "D:\Project\ViT_ImageNet\Train"

class_names = sorted(os.listdir(dataset_path))

print(class_names)

with open("imagenet_test_classes.py", "w") as f:
    f.write(f"class_labels = {class_names}\n")


print("Class names saved in imagenet_test_classes.py")