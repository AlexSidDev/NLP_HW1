import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_scores(model_predictions: dict[str, np.array], y_true: np.array):

  scores = {}
  for model, preds in model_predictions.items():
    scores[model] = {
        'f1': f1_score(y_true, preds, average='macro'),
        'recall': recall_score(y_true, preds, average='macro'),
        'precision': precision_score(y_true, preds, average='macro'),
    }
  data = pd.DataFrame.from_dict(scores, orient='index')
  data.plot.bar(rot=45, title='Scores comparison')

def plot_roc_auc(model_predictions: dict[str, np.array], y_true: np.array):
  class_names = ['neutral', 'negative', 'positive']
  classes = [0, 1, 2]
  y_true_binary = label_binarize(y_true, classes=classes)

  num_models = len(model_predictions)
  num_classes = len(classes)
  fig, axes = plt.subplots(num_models, num_classes, figsize=(15, num_models * 3))

  for i, (model_name, preds) in enumerate(model_predictions.items()):
      preds_binary = label_binarize(preds, classes=classes)
      axes[i, 0].set_ylabel(f'{model_name}')
      for j, class_name in zip(classes, class_names):
          fpr, tpr, _ = roc_curve(y_true_binary[:, j], preds_binary[:, j])
          roc_auc = auc(fpr, tpr)

          ax = axes[i, j] if num_models > 1 else axes[j]
          ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
          ax.plot([0, 1], [0, 1], 'k--', lw=2)
          ax.set_xlim([0.0, 1.0])
          ax.set_ylim([0.0, 1.05])
          ax.legend(loc="lower right")
          axes[0, j].set_title(f'{class_name}')


  fig.text(0.5, 0.04, '\n\nFalse Positive Rate', ha='center', size=16)
  fig.text(0.04, 0.5, 'True Positive Rate\n\n', va='center', rotation='vertical', size=16)

  plt.tight_layout(rect=[0.04, 0.04, 1, 1])
  plt.show()