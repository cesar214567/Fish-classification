import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = np.repeat(np.arange(0,10),15)

class_names = np.array(['ALB','BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT'])



def plot_confusion_matrix(confusion_matrix_file, cmap=plt.cm.Blues):
    model = confusion_matrix_file.split("_")[0]
    title = model + " conf matrix"


    # Compute confusion matrix
    cm = np.transpose(np.loadtxt(confusion_matrix_file,
                 delimiter=",", dtype=float))
    # Only use the labels that appear in the data
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' 
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(f'{model}_conf_matrix.jpg')
    plt.clf()
    return ax


np.set_printoptions(precision=2)

confusion_matrixes = ["UniDet_conf_matrix.csv",
                      "yolo_conf_matrix.csv",
                      "yoloTrained3_conf_matrix.csv",
                      "experimento3_conf_matrix.csv",
                      "experimento2_conf_matrix.csv"]
for i in confusion_matrixes:
   plot_confusion_matrix(i)
