import numpy as np 
import matplotlib.pyplot as plt 
import json

RESULT_PATHS = [
    # './results/With Pertribations/hands_cropped.json',
    # # './results/With Pertribations/hands_full_body.json',
    # './results/With Pertribations/hands_underwaterizer_v1_epoch_5.json',
    # # './results/With Pertribations/hands_underwaterizer_v1_epoch_10.json',
    # # './results/With Pertribations/hands_underwaterizer_v1_epoch_15.json',
    # # './results/With Pertribations/cycle_v1_epoch_10.json',
    # # './results/With Pertribations/cycle_v1_epoch_25.json',
    # './results/With Pertribations/VAROS_v1_epoch_35.json',
    # './results/With Pertribations/VAROS_v1_epoch_45.json',
    # './results/With Pertribations/VAROS_v1_epoch_55.json',
    # './results/With Pertribations/VAROS_v1_epoch_65.json',
    # './results/With Pertribations/VAROS_depth_v1_epoch_5.json',
    # './results/With Pertribations/VAROS_depth_v1_epoch_30.json',
    # './results/With Pertribations/VAROS_depth_v1_epoch_40.json',
    # './results/With Pertribations/hands_underwaterizer_v1_epoch_15_CHEAT.json',
    './results/res18_hands_cropped.json',
    './results/res18_hands_underwaterizer_v1_epoch_5.json',
    './results/res18_hands_underwaterizer_v1_epoch_10.json',
    './results/res18_hands_underwaterizer_v1_epoch_15.json',
    './results/res18_VAROS_v1_epoch_35.json',
    './results/res18_VAROS_v1_epoch_45.json',
    './results/res18_VAROS_v1_epoch_55.json',
    './results/res18_VAROS_v1_epoch_65.json',
    './results/res18_VAROS_depth_v1_epoch_5.json',
    './results/res18_VAROS_depth_v1_epoch_30.json',
    './results/res18_VAROS_depth_v1_epoch_40.json',
    './results/res18_hands_underwaterizer_v1_epoch_15_CHEAT.json',
]

# Read in files
model_results = dict()
for path in RESULT_PATHS:
    with open(path, 'r') as json_file:
        json_data = json.load(json_file)
        model_results[json_data['name']] = json_data

# Parse results getting the recall by class
names = []
results = dict()
for result_name, result_data in model_results.items():
    names.append(result_name)

    for class_name, recall in result_data['recall_test'].items():

        class_data = results.get(class_name)
        if class_data is None:
            results[class_name] = [recall]
        else:
            results[class_name].append(recall)


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
plt.yticks([-.005, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])

plt.legend()
plt.show() 
