import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

PATH_STATS = 'stats/Stillberg2_bs_8_LR_0.01_epochs_15_weighted_True.json'

X_epochs = []
Y_epoch_loss = []
Y_accuracy = []
Y_precision = []
Y_recall = []
Y_f1 = []

# read json file
def open_json(fname):
    with open(fname) as f:
        nested_dict = json.load(f)
    return nested_dict

nested_dict = open_json(PATH_STATS)

for key in nested_dict.keys():
    if not 'time' in key:
        X_epochs.append(key)
        Y_epoch_loss.append(nested_dict[key]['epoch_loss'])
        Y_accuracy.append(nested_dict[key]['accuracy'])
        Y_precision.append(nested_dict[key]['precision'])
        Y_recall.append(nested_dict[key]['recall'])
        Y_f1.append(nested_dict[key]['f1'])

print()

# plot decoding
plt.title('Train Stats')
plt.xlabel('Epochs')
plt.ylabel('Training Statistics')
plt.plot(X_epochs, Y_accuracy, label="Accuracy")
plt.plot(X_epochs, Y_precision, label="Precision")
plt.plot(X_epochs, Y_recall, label="Recall")
plt.plot(X_epochs, Y_f1, label="F1-score")
plt.xticks(rotation=90)

plt.legend(loc="upper left")
plt.savefig(f'stats/plots/{PATH_STATS[6:-5]}_stats.png')
plt.show()
plt.clf()
plt.cla()
plt.close()
### 

plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Training Statistics')
plt.plot(X_epochs, Y_epoch_loss, label="Loss (averaged per epoch)")
plt.xticks(rotation=90)

plt.legend(loc="upper left")
plt.savefig(f'stats/plots/{PATH_STATS[6:-5]}_loss.png')
plt.show()
