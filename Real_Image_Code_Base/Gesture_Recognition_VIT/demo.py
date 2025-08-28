from datasets import load_dataset
import matplotlib.pyplot as plt
from datasets import DatasetDict
from transformers import ViTImageProcessor
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Resize
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from transformers import TrainingArguments, Trainer
import numpy as np

dataset = load_dataset("./datasets/hands_cropped")


train_val_split = dataset["train"].train_test_split(test_size=0.2)

dataset = DatasetDict({
    "train": train_val_split["train"],
    "validation": train_val_split["test"],
    "test": dataset["test"]
})
print(dataset)

train_ds = dataset['train']
val_ds = dataset['validation']
test_ds = dataset['test']




# Loop through the dataset and plot the first image of each label
shown_labels = set()
plt.figure(figsize=(10, 10))
for i, sample in enumerate(train_ds):
    label = train_ds.features['label'].names[sample['label']]
    if label not in shown_labels:
        plt.subplot(1, len(train_ds.features['label'].names), len(shown_labels) + 1)
        plt.imshow(sample['image'])
        plt.title(label)
        plt.axis('off')
        shown_labels.add(label)
        if len(shown_labels) == len(train_ds.features['label'].names):
            break

plt.show()


id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}
id2label, id2label[train_ds[0]['label']]

model_name = "google/vit-large-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
print(processor)
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)

transforms = Compose([
    #RandomResizedCrop(size),
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize,
])


def apply_transforms(examples):
    examples['pixel_values'] = [transforms(image.convert("RGB")) for image in examples['image']]
    return examples


train_ds.set_transform(apply_transforms)
val_ds.set_transform(apply_transforms)
test_ds.set_transform(apply_transforms)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)
for batch in train_dl:
    batch
    pixes = batch['pixel_values']
    labels = batch['labels']
    for pix, label in zip(pixes, labels):
        plt.imshow(pix.permute(1, 2, 0))
        plt.title(f"Label: {id2label[label.item()]}")
        plt.show()

counters = dict()
for batch in train_dl:
    batch
    pixes = batch['pixel_values']
    labels = batch['labels']
    for pix, label_tensor in zip(pixes, labels):
        label_id = label_tensor.item()
        label_counter = counters.get(id2label[label_id])
        if label_counter is None:
            counters[id2label[label_id]] = 1
        else:
            counters[id2label[label_id]] += 1


print(counters)