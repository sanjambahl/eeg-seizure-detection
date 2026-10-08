"""Tasks, models and evaluation.

The key idea: always test on RECORDINGS the model has never seen.
`StratifiedGroupKFold` keeps every window from one recording together, so a
recording is either entirely in training or entirely in testing.
"""

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# name -> (non-seizure / class-0 sets, class-1 sets, short description)
# A task with class-1 = None is multi-class over the listed sets.
TASKS = {
    "seizure_vs_healthy": (["Z"], ["S"], "Seizure vs healthy (A vs E)"),
    "seizure_vs_all": (["Z", "O", "N", "F"], ["S"], "Seizure vs all non-seizure EEG"),
    "seizure_vs_interictal": (["N", "F"], ["S"], "Seizure vs same patients between seizures"),
    "healthy_vs_epilepsy": (["Z", "O"], ["N", "F"], "Healthy vs epilepsy between seizures"),
    "zone_vs_opposite": (["N"], ["F"], "Seizure zone vs opposite side, between seizures"),
    "five_class": (["Z", "O", "N", "F", "S"], None, "All five groups"),
}

TASK_LABELS = {
    "seizure_vs_healthy": "Seizure vs healthy (A vs E)",
    "seizure_vs_all": "Seizure vs all non-seizure",
    "seizure_vs_interictal": "Seizure vs same patients, no seizure",
    "healthy_vs_epilepsy": "Healthy vs epilepsy, no seizure",
    "zone_vs_opposite": "Seizure zone vs opposite side",
    "five_class": "All five groups",
}


def make_models(seed=0):
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")
        ),
        "SVM (RBF kernel)": make_pipeline(
            StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced")
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=seed
        ),
        "Neural network (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True,
                          random_state=seed),
        ),
    }


def task_data(task, X, window_sets, groups):
    """Select the windows used by a task and build its labels."""
    neg, pos, _ = TASKS[task]
    if pos is None:  # multi-class
        mask = np.isin(window_sets, neg)
        y = np.array([neg.index(s) for s in window_sets[mask]])
    else:
        mask = np.isin(window_sets, neg + pos)
        y = np.isin(window_sets[mask], pos).astype(int)
    return X[mask], y, groups[mask], window_sets[mask]


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if p.shape[1] == 2 else p
    return model.decision_function(X)


def cross_validate(model, X, y, groups, split="grouped", n_splits=5, seeds=(0, 1, 2)):
    """Repeated k-fold cross-validation.

    split="grouped": windows from one recording never appear in both train and test (honest).
    split="leaky":   windows are shuffled individually, so the same recording leaks into both.

    Returns a dict of per-fold metric lists plus the pooled confusion matrix.
    """
    binary = len(np.unique(y)) == 2
    out = {"balanced_accuracy": [], "sensitivity": [], "specificity": [], "auc": [],
           "recording_balanced_accuracy": []}
    n_classes = len(np.unique(y))
    cm_total = np.zeros((n_classes, n_classes), dtype=int)

    for seed in seeds:
        if split == "grouped":
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            folds = cv.split(X, y, groups)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            folds = cv.split(X, y)
        for train, test in folds:
            m = clone(model)
            if "random_state" in m.get_params():
                m.set_params(random_state=seed)
            m.fit(X[train], y[train])
            pred = m.predict(X[test])
            out["balanced_accuracy"].append(balanced_accuracy_score(y[test], pred))
            cm_total += confusion_matrix(y[test], pred, labels=range(n_classes))
            if binary:
                tn, fp, fn, tp = confusion_matrix(y[test], pred, labels=[0, 1]).ravel()
                out["sensitivity"].append(tp / (tp + fn))
                out["specificity"].append(tn / (tn + fp))
                out["auc"].append(roc_auc_score(y[test], _scores(m, X[test])))
            # Recording-level decision: majority vote over that recording's windows
            if split == "grouped":
                g_test = groups[test]
                rec_true, rec_pred = [], []
                for g in np.unique(g_test):
                    idx = g_test == g
                    rec_true.append(y[test][idx][0])
                    rec_pred.append(np.bincount(pred[idx], minlength=n_classes).argmax())
                out["recording_balanced_accuracy"].append(
                    balanced_accuracy_score(rec_true, rec_pred)
                )
    out = {k: v for k, v in out.items() if v}
    out["confusion_matrix"] = cm_total
    return out


def detector_transfer(model, X, window_sets, groups, train_neg_sets, n_splits=5, seeds=(0, 1, 2)):
    """Train a seizure detector on `train_neg_sets` vs seizure, then ask:
    how often does it shout "seizure" on held-out recordings from EVERY group?

    Every model sees the same held-out recordings, so detectors trained on
    different data can be compared fairly. Returns {set letter: list of
    per-fold 'predicted seizure' rates}.
    """
    y_all = (window_sets == "S").astype(int)
    rates = {s: [] for s in ["Z", "O", "N", "F", "S"]}
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        strat = np.searchsorted(["F", "N", "O", "S", "Z"], window_sets)  # stratify by group
        for train, test in cv.split(X, strat, groups):
            use = train[np.isin(window_sets[train], train_neg_sets + ["S"])]
            m = clone(model)
            if "random_state" in m.get_params():
                m.set_params(random_state=seed)
            m.fit(X[use], y_all[use])
            pred = m.predict(X[test])
            for s in rates:
                idx = window_sets[test] == s
                rates[s].append(pred[idx].mean())
    return rates
