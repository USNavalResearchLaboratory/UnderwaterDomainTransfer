from datasets import load_dataset
import matplotlib.pyplot as plt
from datasets import DatasetDict
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Resize, InterpolationMode, RandomRotation, v2
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, ViTForImageClassification, ViTImageProcessor, ResNetModel, ResNetForImageClassification, AutoImageProcessor, AutoConfig
from transformers.trainer_callback import EarlyStoppingCallback
from sklearn.metrics import recall_score, confusion_matrix, ConfusionMatrixDisplay
import argparse
import json
from datetime import datetime
import os
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, help='The path to the dataset file')
parser.add_argument('--name', type=str, help='A name for the model, used for output files')
parser.add_argument('--n_epochs', type=int, help='How many epochs to train for, early stopping is always active!')
parser.add_argument('--n_patience', type=int, help='Patience for early stopping')
parser.add_argument('--batch_size', type=int, help='Batch Size')
args = parser.parse_args()

START_TIME = datetime.now()

DATASET_PATH = args.dataset
NAME = args.name
N_EPOCHS = args.n_epochs
N_PATIENCE = args.n_patience
BATCH_SIZE = args.batch_size

dataset = load_dataset(DATASET_PATH)


train_val_split = dataset["train"].train_test_split(test_size=0.2, shuffle=True)

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
# shown_labels = set()
# plt.figure(figsize=(10, 10))
# for i, sample in enumerate(train_ds):
#     label = train_ds.features['label'].names[sample['label']]
#     if label not in shown_labels:
#         plt.subplot(1, len(train_ds.features['label'].names), len(shown_labels) + 1)
#         plt.imshow(sample['image'])
#         plt.title(label)
#         plt.axis('off')
#         shown_labels.add(label)
#         if len(shown_labels) == len(train_ds.features['label'].names):
#             break
# plt.show()


id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}
id2label, id2label[train_ds[0]['label']]

model_name = "microsoft/resnet-18"
processor = AutoImageProcessor.from_pretrained(model_name)
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["shortest_edge"]

normalize = Normalize(mean=image_mean, std=image_std)

transforms = Compose([
    # RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=InterpolationMode.BILINEAR, antialias = True),
    # RandomRotation((-15, 15), interpolation=InterpolationMode.BILINEAR, expand=False, center=None, fill=0),
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize,
    # v2.GaussianNoise(),
])


test_transforms = Compose([
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize,
])

def apply_transforms(examples):
    examples['pixel_values'] = [transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def apply_test_transforms(examples):
    examples['pixel_values'] = [test_transforms(image.convert("RGB")) for image in examples['image']]
    return examples


train_ds.set_transform(apply_transforms)
val_ds.set_transform(apply_test_transforms)
test_ds.set_transform(apply_test_transforms)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=BATCH_SIZE)
# for batch in train_dl:
#     batch
#     pixes = batch['pixel_values']
#     labels = batch['labels']
#     for pix, label in zip(pixes, labels):
#         plt.imshow(pix.permute(1, 2, 0))
#         plt.title(f"Label: {id2label[label.item()]}")
#         plt.show()



# model = ResNetForImageClassification.from_pretrained(model_name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
# model = ResNetForImageClassification(AutoConfig.from_pretrained(model_name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True))
model = ResNetForImageClassification.from_pretrained(model_name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
os.makedirs(f'output-models/{NAME}', exist_ok=True)
train_args = TrainingArguments(
    output_dir = f'output-models/{NAME}',
    overwrite_output_dir= True,
    save_total_limit=2,
    report_to="tensorboard",
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=N_EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir='logs',
    remove_unused_columns=False,
)
trainer = Trainer(
    model,
    train_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    tokenizer=processor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=N_PATIENCE)]
)
trainer.train()

train_outputs = trainer.predict(train_ds)
val_outputs = trainer.predict(val_ds)
outputs = trainer.predict(test_ds)

def make_cm(outputs, set_name):
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)

    labels = train_ds.features['label'].names
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)
    plt.savefig(f'./vis/{NAME}_{set_name}.png', dpi=300)

make_cm(train_outputs, 'train')
make_cm(val_outputs, 'val')
make_cm(outputs, 'test')

# Calculate the recall scores
END_TIME = datetime.now()
recall_store = {
    'name': NAME,
    'epochs trained': trainer.state.epoch,
    'start_time': START_TIME.strftime("%Y-%m-%d %H:%M:%S"),
    'end_time': END_TIME.strftime("%Y-%m-%d %H:%M:%S"),
    'result_metric': outputs.metrics,
}

def make_recall(outputs, set_name):
    labels = train_ds.features['label'].names
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)
    recall = recall_score(y_true, y_pred, average=None)

    recall_store['recall_' + set_name] = dict()
    for label, score in zip(labels, recall):
        recall_store['recall_' + set_name][label] = score

make_recall(train_outputs, 'train')
make_recall(val_outputs, 'val')
make_recall(outputs, 'test')

with open(f'./results/{NAME}.json', "w") as json_file:
    json.dump(recall_store, json_file, indent=1)

