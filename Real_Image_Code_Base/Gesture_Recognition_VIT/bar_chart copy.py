import numpy as np 
import matplotlib.pyplot as plt 
import json
from pprint import pprint


ROOT_PATH = './results'
MODEL_PATHS = [
    'hands_cropped.json',
    'hands_full_body.json',
    'hands_underwaterizer_v1_epoch_5.json',
    'hands_underwaterizer_v1_epoch_10.json',
    'hands_underwaterizer_v1_epoch_15.json',
    'cycle_v1_epoch_10.json',
    'cycle_v1_epoch_25.json',
    'hands_underwaterizer_v1_epoch_15_CHEAT.json',
]
RUNS_PATHS = [
    'Run 1',
    'Run 2',
    'Run 3',
    'Run 4',
    'Run 5',
]

# Read in files
model_results = dict()
for run in RUNS_PATHS:
    for model in MODEL_PATHS:
        full_path = f'{ROOT_PATH}/{run}/{model}'
        with open(full_path, 'r') as json_file:
            json_data = json.load(json_file)

            if model_results.get(run) is None:
                model_results[run] = dict()

            model_results[run][json_data['name']] = json_data

pprint(model_results)

mean_results = dict()
for run in RUNS_PATHS:
    for model_name, model_data in model_results[run].items():
        if mean_results.get(model_name) is None:
            mean_results[model_name] = dict()

        for class_name, recall in model_data['recall_test'].items():
            if mean_results[model_name].get(class_name) is None:
                mean_results[model_name][class_name] = [recall]
            else:
                mean_results[model_name][class_name].append(recall)

pprint(mean_results)

for model_name, model_data in mean_results.items():
    for class_name, recalls in mean_results[model_name].items():
        mean_results[model_name][class_name] = sum(recalls) / len(recalls)

pprint(mean_results)

# Parse results getting the recall by class
names = []
results = dict()
for model_name, model_data in mean_results.items():
    names.append(model_name)

    for class_name, recall in mean_results[model_name].items():

        class_data = results.get(class_name)
        if class_data is None:
            results[class_name] = [recall]
        else:
            results[class_name].append(recall)

pprint(results)

amnt_of_classes = len(list(results.keys()))
amnt_of_models = len(names)

fig = plt.subplots(figsize =(20, 8)) 

# Plot each class
color = iter(plt.cm.Pastel1(np.linspace(0, 1, amnt_of_classes)))
index = 0
barWidth = 0.10
br = np.arange(amnt_of_models)
for class_name, recall_results in results.items():
    plt.bar(br, recall_results, color =next(color), width = barWidth, edgecolor ='grey', label = class_name) 

    br = [x + barWidth for x in br]
    index += 1


plt.xlabel('Models', fontweight ='bold', fontsize = 15) 
plt.ylabel('Recall', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(names))], names, rotation=-15)
plt.yticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])

plt.legend()
plt.show() 
