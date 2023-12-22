from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, linear_model, metrics
from sklearn.metrics import roc_auc_score

RANDOM_SEED = 7

def lr_train(X_train, y_train):
    """ Tuning hyperparameters of Logistic Regression
    Args:
        X_train: training data
        y_train: training label
    Return:
        df_results: dataframe of hyperparameter tuning results
        clf: best model
    """

    hyperparams_dict = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10],
        'class_weight': [None, 'balanced']}
    max_iter = 500

    param_sets = list(model_selection.ParameterGrid(hyperparams_dict))
    df_results = pd.DataFrame([], index=np.arange(len(param_sets)),
                            columns=list(hyperparams_dict.keys()) + ['score_mean', 'score_std'])

    for idx, param_set in tqdm(enumerate(param_sets), total=len(param_sets)):
        C = 1 if param_set['C'] is None else param_set['C']
        penalty = None if C==0 else param_set['penalty']      
        class_weight = param_set['class_weight']
        solver = 'lbfgs' if penalty==None or penalty=='l2' else 'liblinear'
        
        clf = LogisticRegression(penalty=penalty, C=C, class_weight=class_weight,
                                            solver=solver, random_state=RANDOM_SEED,
                                            max_iter=max_iter)
        
        clf.fit(X_train, y_train)

        for key, val in param_set.items():
            if key=='penalty':
                # Reecord actual penalty.
                df_results.at[idx, key] = penalty
            else:
                df_results.at[idx, key] = val
        
        # tune based on AUC. Ideally should use val set
        scores = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1]) 

        df_results.at[idx, 'score_mean'] = scores.mean()
        df_results.at[idx, 'score_std'] = scores.std()

    # Get best models using best parameters
    best_params = df_results.sort_values(by='score_mean', ascending=False).iloc[0]

    penalty_best = best_params.penalty
    C_best = best_params.C
    class_weight_best = best_params.class_weight
    solver = 'lbfgs' if penalty_best==None or penalty_best=='l2' else 'liblinear'
    
    clf = linear_model.LogisticRegression(penalty=penalty_best, C=C_best, class_weight=class_weight_best,
                                      solver=solver, random_state=random.randint(0, 10000),
                                      max_iter=max_iter)
    
    return df_results, clf

def rf_train(X_train, y_train):
    """ Tuning hyperparameters of Random Forest
    Args:
        X_train: training data
        y_train: training label
    Return:
        df_results: dataframe of hyperparameter tuning results
        clf: best model
    """

    # Define hyperparameters for Random Forest
    hyperparams_dict = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'class_weight': [None, 'balanced']
    }

    num_cv = 5

    param_sets = list(model_selection.ParameterGrid(hyperparams_dict))
    df_results = pd.DataFrame([], index=np.arange(len(param_sets)),
                            columns=list(hyperparams_dict.keys()) + ['score_mean', 'score_std'])
    
    # Iterate through each parameter set
    for idx, param_set in tqdm(enumerate(param_sets), total=len(param_sets)):
        clf = RandomForestClassifier(n_estimators=param_set['n_estimators'],
                                    max_depth=param_set['max_depth'],
                                    min_samples_split=param_set['min_samples_split'],
                                    class_weight=param_set['class_weight'],
                                    random_state=RANDOM_SEED)

        clf.fit(X_train, y_train)

        # Record parameters and their scores
        for key, val in param_set.items():
            df_results.at[idx, key] = val

        scores = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])

        df_results.at[idx, 'score_mean'] = scores.mean()
        df_results.at[idx, 'score_std'] = scores.std()

    best_params = df_results.sort_values(by='score_mean', ascending=False).iloc[0]

    param_set = {
        'n_estimators': best_params.n_estimators,
        'max_depth': best_params.max_depth,
        'min_samples_split': best_params.min_samples_split,
        'class_weight': best_params.class_weight
    }

    clf = RandomForestClassifier(n_estimators=param_set['n_estimators'],
                                max_depth=param_set['max_depth'],
                                min_samples_split=param_set['min_samples_split'],
                                class_weight=param_set['class_weight'],
                                random_state=RANDOM_SEED)
    return df_results, clf