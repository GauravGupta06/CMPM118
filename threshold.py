import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression



# Compute the metrics
def evalute(y_test, y_pred, threshold):
    y_pred = (y_pred[:,1] >= threshold).astype(bool)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred)}')
    print(f'Recall: {recall_score(y_test, y_pred)}')
    print(f'F1: {f1_score(y_test, y_pred)}')

# Plot ROC curves
def plot_roc(fpr, tpr, thresholds, g_opt_idx=None, j_opt_idx=None):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
    plt.plot([0,1], [0,1], linestyle='--', label='Random Classifier')
    plt.plot(fpr, tpr, linewidth=2, label='Logistic')
    if g_opt_idx:
        plt.scatter(fpr[g_opt_idx], tpr[g_opt_idx], s=15, color='green', label='G-mean Optimal Threshold', zorder=5)
        threshold = thresholds[g_opt_idx]
        print(f'G-mean Optimal Threshold: {threshold}')
    elif j_opt_idx:
        plt.scatter(fpr[j_opt_idx], tpr[j_opt_idx], s=15, color='green', label='J statistic Optimal Threshold', zorder=5)
        threshold = thresholds[j_opt_idx]
        print(f"Youden's J statistic Optimal Threshold: {threshold}")
    else:
        threshold = 0.5
        print(f"Default Threshold: {0.5}")
    evalute(y_test, y_pred, threshold)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    return threshold

def getG_Thresh(X,y, plot=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])

    g_mean = np.sqrt(tpr * (1-fpr))
    g_opt_idx = np.argmax(g_mean)
    g_thresh = plot_roc(fpr, tpr, thresholds, g_opt_idx=g_opt_idx)

    print(f"G-mean: {g_thresh:.3f}")

    evaluate(y_test, y_pred, g_thresh)

    return g_thresh

def main():
    X, y = make_classification(
    n_samples=10000, 
    n_redundant=0,
    n_clusters_per_class=1, 
    weights=[0.9],
    flip_y=0, 
    random_state=24
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
    plot_roc(fpr, tpr, thresholds)  

    j_opt_idx = np.argmax(tpr-fpr)
    plot_roc(fpr, tpr, thresholds, j_opt_idx=j_opt_idx) 

    g_mean = np.sqrt(tpr * (1-fpr))
    g_opt_idx = np.argmax(g_mean)
    plot_roc(fpr, tpr, thresholds, g_opt_idx=g_opt_idx) 