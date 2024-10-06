from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import auc
from tqdm import tqdm


def validate_model(df_train, df_val, model, param_list):
    """Calculates the validation performance for the set of parameters for NRE or Flow-based alternative"""
    ml_models = {'Linear Support Vector Machines': LinearSVC(dual='auto'), 'Decision Tree': DecisionTreeClassifier(),
                 'Random Forest': RandomForestClassifier(), 'Naive Bayes': GaussianNB()}

    auc_scores = []
    for params in tqdm(param_list):
        roc_curves = {}
        _ = model(df_train, ml_models, test_df=df_val, roc_curves=roc_curves, **params)

        max_auc = 0
        for mdl in ml_models:
            curr_auc = auc(roc_curves[mdl][0], roc_curves[mdl][1])
            if curr_auc > max_auc:
                max_auc = curr_auc
        auc_scores.append((params, max_auc))
    return auc_scores


