import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def roc_sweep(params, y_act, X, typ='pessimistic'):
    """
    Sweeps through the params to form the ROC curve points
    """
    TPRs = {"Robust covariance": [], "One-Class SVM": [], "Isolation Forest": [],
            "Local Outlier Factor": []}
    FPRs = {"Robust covariance": [], "One-Class SVM": [], "Isolation Forest": [],
            "Local Outlier Factor": []}

    for outliers_fraction in params:

        anomaly_algorithms = [
            ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
            ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)),
            ("Isolation Forest",
             IsolationForest(contamination=outliers_fraction, random_state=42),
             ),
            ("Local Outlier Factor",
             LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction),
             ),
        ]
        for name, algorithm in anomaly_algorithms:
            algorithm.fit(X)
            if name == "Local Outlier Factor":
                y_pred = algorithm.fit_predict(X)
            else:
                y_pred = algorithm.fit(X).predict(X)
            TP, FP, TN, FN = perf_measure(y_act, y_pred)
            if TP == FN == 0 or FP == TN == 0:
                continue
            TPRs[name].append(TP / (TP + FN))
            FPRs[name].append(FP / (FP + TN))

    for name, TPR, FPR in zip(TPRs.keys(), TPRs.values(), FPRs.values()):
        TPR, FPR = add_disc_points(TPR, FPR, typ=typ)
        plt.plot(FPR, TPR, label=name)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.title('ROC of Anomaly Detection with ' + conn_param)

    return TPRs, FPRs