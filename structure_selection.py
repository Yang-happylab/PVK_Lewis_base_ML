"""Fold-local feature selection and nested evaluation for structure models."""

import json
import logging
from time import perf_counter
from types import SimpleNamespace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.feature_selection import RFE, VarianceThreshold
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, ParameterSampler, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CV_CONTRACT = {
    'method': 'NestedRepeatedStratifiedKFold', 'n_splits': 4, 'n_repeats': 3,
    'requested_strata': 4, 'random_state': 42, 'inner_splits': 4,
    'inner_repeats': 3, 'max_selection_paths': 2,
    'evaluation_scope': 'outer_training_nested_cv',
    'selection_rule': 'one_standard_error_then_fewer_features',
    'search_strategy': 'repeated_tune_multipath_retune',
}


def evaluation_cv_contract(manifest):
    """Compare evaluation rules independently of parameter-search resources."""
    return {key: value for key, value in manifest.items()
            if key != 'parameter_candidates_per_stage_max'}


def cv_splits(y, n_splits=4, repeats=1, seed=42):
    strata = pd.qcut(np.asarray(y), q=4, labels=False, duplicates='drop')
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats,
                                     random_state=seed)
    return list(splitter.split(np.zeros(len(y)), strata))


def sample_candidates(params, budget, seed=42):
    """Sample unique grid values with equal budgets for conditional branches."""
    branches = params if isinstance(params, list) else [params]
    candidates = []
    for index, branch in enumerate(branches):
        grid = {key: list(dict.fromkeys(values)) for key, values in branch.items()}
        count = budget // len(branches) + (index < budget % len(branches))
        if count:
            candidates.extend(ParameterSampler(grid, n_iter=count, random_state=seed + index))
    return [{key: [value] for key, value in candidate.items()} for candidate in candidates]


class SelectedRegressor(RegressorMixin, BaseEstimator):
    """All preprocessing and selection are fitted on this call's training rows."""

    def __init__(self, estimator, n_features=32, method='rfe', inner_splits=4, seed=42):
        self.estimator = estimator
        self.n_features = n_features
        self.method = method
        self.inner_splits = inner_splits
        self.seed = seed

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        variance = VarianceThreshold().fit(X)
        available = np.flatnonzero(variance.get_support())
        count = min(self.n_features, len(available))
        base = Pipeline([('scaler', StandardScaler()), ('model', clone(self.estimator))])
        if count == len(available):
            chosen = available
        else:
            importance = ('feature_importances_' if hasattr(self.estimator, 'feature_importances_')
                          or self.estimator.__class__.__name__ in ('RandomForestRegressor', 'XGBRegressor')
                          else 'coef_')
            selector = RFE(base, n_features_to_select=count, step=1,
                           importance_getter='named_steps.model.' + importance).fit(X[:, available], y)
            chosen = available[selector.get_support()]
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[chosen] = True
        self.pipeline_ = base.fit(X[:, chosen], y)
        return self

    def predict(self, X):
        return self.pipeline_.predict(np.asarray(X)[:, self.support_])


def score_summary(values, n_splits, repeats):
    """Repeat means set the one-SE heuristic; folds are not independent replicates."""
    values = np.asarray(values)
    repeat_means = values.reshape(-1, repeats, n_splits).mean(axis=2)
    se = (repeat_means.std(axis=1, ddof=1) / np.sqrt(repeats) if repeats > 1
          else values.std(axis=1) / np.sqrt(n_splits))
    return {'mean_test_r2': values.mean(axis=1), 'std_test_r2': values.std(axis=1),
            'se_test_r2': se}


def parameter_paths(results, n_splits, repeats, limit):
    values = np.column_stack([results[f'split{i}_test_score']
                              for i in range(n_splits * repeats)])
    summary = score_summary(values, n_splits, repeats)
    means = summary['mean_test_r2']
    best = int(np.argmax(means))
    eligible = np.flatnonzero(means >= means[best] - summary['se_test_r2'][best])
    ignored = {'tol', 'max_iter', 'random_state', 'selection', 'compute_score', 'n_jobs', 'verbosity'}
    selected, seen = [], set()
    for i in sorted(eligible, key=lambda i: (-means[i], summary['std_test_r2'][i], i)):
        params = {k.removeprefix('model__'): v for k, v in results['params'][i].items()}
        unused = ({'gamma', 'degree', 'coef0'} if params.get('kernel') == 'linear' else
                  {'degree', 'coef0'} if params.get('kernel') == 'rbf' else set())
        signature = repr(sorted((k, v) for k, v in params.items() if k not in ignored | unused))
        if signature not in seen:
            selected.append(int(i)); seen.add(signature)
        if len(selected) == limit:
            break
    return selected


def one_se_index(results, n_splits):
    mean = np.asarray(results['mean_test_r2'])
    std = np.asarray(results['std_test_r2'])
    best = int(np.argmax(mean))
    se = results.get('se_test_r2', std / np.sqrt(n_splits))
    threshold = mean[best] - se[best]
    eligible = np.flatnonzero(mean >= threshold)
    return min(eligible, key=lambda i: (-results['mean_test_size'][i], -mean[i], std[i], i))


def nested_select(X, y, feature_names, names, estimator, params, n_iter, output_dir,
                  method='rfe', outer_splits=4, repeats=3, inner_splits=4,
                  seed=42, n_jobs=-1, inner_repeats=3, max_paths=2):
    """No external-test argument: evaluate selection, then refit on all training rows."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    names, feature_names = np.asarray(names), np.asarray(feature_names)
    output = Path(output_dir) / 'feature_selection'
    output.mkdir(parents=True, exist_ok=True)
    budget = int(n_iter)
    if budget < 1:
        raise ValueError('Parameter search budget must be positive')
    parameter_candidates = sample_candidates(params, budget, seed)
    fold_rows, predictions, selections, candidate_tables = [], [], [], []

    def select(a, stage):
        started = perf_counter()
        inner = cv_splits(y[a], inner_splits, inner_repeats, seed)
        def tune(cols, phase):
            logging.info('%s %s: tuning at most %d candidates', stage, phase, budget)
            pipeline = Pipeline([('variance', VarianceThreshold()),
                                 ('scaler', StandardScaler()), ('model', clone(estimator))])
            search = GridSearchCV(
                pipeline, parameter_candidates, scoring='r2', cv=inner,
                refit=True, n_jobs=n_jobs, error_score='raise',
            ).fit(X[a][:, cols], y[a])
            results = search.cv_results_
            summary = score_summary(np.column_stack([results[f'split{i}_test_score']
                                     for i in range(len(inner))]), inner_splits, inner_repeats)
            table = pd.DataFrame({
                'Stage': stage, 'Phase': phase, 'Candidate': np.arange(len(results['params'])),
                'R2_mean': results['mean_test_score'], 'R2_SD': results['std_test_score'],
                'R2_SE_repeat_means': summary['se_test_r2'],
                'Mean_fit_seconds': results['mean_fit_time'],
                'Selected': np.arange(len(results['params'])) == search.best_index_,
                'Parameters': [json.dumps(p, default=lambda v: v.item()) for p in results['params']],
            })
            candidate_tables.append(table)
            pd.concat(candidate_tables, ignore_index=True).to_csv(output / 'inner_search.csv', index=False)
            return search

        initial = tune(np.arange(X.shape[1]), 'full_tuning')
        available = np.flatnonzero(VarianceThreshold().fit(X[a]).get_support())
        path_indices = parameter_paths(initial.cv_results_, inner_splits, inner_repeats, max_paths)
        tables, path_columns, fixed_estimators = [], {}, {}
        for path_id, index in enumerate(path_indices, 1):
            parameters = initial.cv_results_['params'][index]
            fixed = clone(estimator).set_params(**{k.removeprefix('model__'): v
                                                   for k, v in parameters.items()})
            fixed_estimators[path_id] = fixed
            logging.info('%s: selecting features (%s), path %d/%d',
                         stage, method, path_id, len(path_indices))
            if method == 'backward':
                # Reuse one complete greedy trajectory, including the all-feature baseline.
                base = Pipeline([('scaler', StandardScaler()), ('model', fixed)])
                def evaluate(cols):
                    return cross_val_score(base, X[a][:, cols], y[a], cv=inner,
                                           scoring='r2', n_jobs=n_jobs, error_score='raise')
                paths, values = [available.tolist()], [evaluate(available)]
                while len(paths[-1]) > 1:
                    candidates = [paths[-1][:i] + paths[-1][i+1:] for i in range(len(paths[-1]))]
                    scores = [evaluate(cols) for cols in candidates]
                    best = int(np.argmax([v.mean() for v in scores]))
                    paths.append(candidates[best]); values.append(scores[best])
                    logging.info('%s backward: %d features, selection CV R2=%.4f',
                                 f'{stage} path {path_id}', len(paths[-1]), values[-1].mean())
                results = score_summary(values, inner_splits, inner_repeats)
                results['mean_test_size'] = -np.array([len(cols) for cols in paths])
            else:
                # Each inner fold traverses RFE once, recording every subset size.
                def rfe_fold(train, valid):
                    cols = np.flatnonzero(VarianceThreshold().fit(X[a][train]).get_support()).tolist()
                    scores = {}
                    while cols:
                        base = Pipeline([('scaler', StandardScaler()), ('model', clone(fixed))])
                        base.fit(X[a][train][:, cols], y[a][train])
                        scores[len(cols)] = r2_score(y[a][valid], base.predict(X[a][valid][:, cols]))
                        model = base['model']
                        importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.coef_ ** 2
                        del cols[int(np.argsort(np.ravel(importance))[0])]
                    return scores
                fold_scores = Parallel(n_jobs=n_jobs)(delayed(rfe_fold)(t, v) for t, v in inner)
                counts = np.arange(1, len(available) + 1)
                values = np.array([[scores[min(count, max(scores))] for scores in fold_scores]
                                   for count in counts])
                results = score_summary(values, inner_splits, inner_repeats)
                results['mean_test_size'] = -counts

            if method == 'backward':
                path_columns[path_id] = paths
            table = pd.DataFrame({
                'Path': path_id, 'N_features': -np.asarray(results['mean_test_size']),
                'R2_mean': results['mean_test_r2'], 'R2_SD': results['std_test_r2'],
                'R2_SE_repeat_means': results['se_test_r2'],
                'Parameters': json.dumps(parameters, default=lambda v: v.item()),
            })
            repeat_values = np.asarray(values).reshape(-1, inner_repeats, inner_splits).mean(axis=2)
            for repeat in range(inner_repeats):
                table[f'Repeat_{repeat + 1}_R2'] = repeat_values[:, repeat]
            tables.append(table)
        table = pd.concat(tables, ignore_index=True)
        chosen = one_se_index({'mean_test_r2': table.R2_mean.to_numpy(),
                               'std_test_r2': table.R2_SD.to_numpy(),
                               'se_test_r2': table.R2_SE_repeat_means.to_numpy(),
                               'mean_test_size': -table.N_features.to_numpy()}, inner_splits)
        table['Selected'] = np.arange(len(table)) == chosen
        table.to_csv(output / f'{stage}_selection_path.csv', index=False)
        winner = table.iloc[chosen]
        path_id, count = int(winner.Path), int(winner.N_features)
        if method == 'backward':
            columns = next(cols for cols in path_columns[path_id] if len(cols) == count)
        else:
            selector = SelectedRegressor(fixed_estimators[path_id], n_features=count).fit(X[a], y[a])
            columns = np.flatnonzero(selector.support_).tolist()
        final = tune(columns, 'selected_tuning')
        mask = np.zeros(X.shape[1], dtype=bool)
        columns = np.asarray(columns)[final.best_estimator_['variance'].get_support()]
        mask[columns] = True
        # Save the established selected-column scaler/model interface used by GA.
        fitted = SelectedRegressor(clone(final.best_estimator_['model']))
        fitted.support_ = mask
        fitted.pipeline_ = Pipeline([('scaler', final.best_estimator_['scaler']),
                                     ('model', final.best_estimator_['model'])])
        logging.info('%s completed: %.1f seconds, %d features', stage,
                     perf_counter() - started, mask.sum())
        return SimpleNamespace(best_estimator_=fitted,
                               best_params_=final.best_params_, cv_results_=final.cv_results_, path_id=path_id)

    for k, (a, b) in enumerate(cv_splits(y, outer_splits, repeats, seed)):
        search = select(a, f'outer_{k + 1:02}')
        fitted = search.best_estimator_
        pred = fitted.predict(X[b])
        selected = set(feature_names[fitted.support_])
        selections.append(selected)
        row = {'Repeat': k // outer_splits + 1, 'Fold': k % outer_splits + 1,
               'R2': r2_score(y[b], pred), 'RMSE': np.sqrt(mean_squared_error(y[b], pred)),
               'MAE': mean_absolute_error(y[b], pred), 'N_features': len(selected),
               'Features': '|'.join(feature_names[fitted.support_])}
        fold_rows.append(row)
        predictions.extend({'Repeat': row['Repeat'], 'Fold': row['Fold'], 'Name': names[i],
                            'Score': y[i], 'Prediction': value} for i, value in zip(b, pred))
        pd.DataFrame(fold_rows).to_csv(output / 'outer_fold_metrics.csv', index=False)
        pd.DataFrame(predictions).to_csv(output / 'outer_oof_predictions.csv', index=False)
        logging.info('Nested CV %d/%d: R2=%.4f, retained=%d',
                     k + 1, outer_splits * repeats, row['R2'], len(selected))

    final = select(np.arange(len(y)), 'final')
    fitted = final.best_estimator_
    pd.DataFrame({'Feature': feature_names,
                  'Selection_frequency': [sum(f in s for s in selections) / len(selections)
                                          for f in feature_names],
                  'Final_selected': fitted.support_}).to_csv(output / 'feature_stability.csv', index=False)
    pd.DataFrame([{'Fold_A': i + 1, 'Fold_B': j + 1,
                   'Jaccard': len(selections[i] & selections[j]) / len(selections[i] | selections[j])}
                  for i, j in combinations(range(len(selections)), 2)]).to_csv(
                      output / 'subset_overlap.csv', index=False)
    selected_params = {key: value
                       for key, value in final.best_params_.items() if key.startswith('model__')}
    (output / 'selection.json').write_text(json.dumps({
        'method': method, 'cv_contract': dict(CV_CONTRACT, n_splits=outer_splits,
                                             n_repeats=repeats, inner_splits=inner_splits,
                                             inner_repeats=inner_repeats, max_selection_paths=max_paths,
                                             random_state=seed),
        'search_candidates_per_stage': budget, 'features': feature_names[fitted.support_].tolist(),
        'sampled_candidates_per_stage': len(parameter_candidates),
        'parameter_sampling': 'unique_values_equal_conditional_branch_budgets',
        'best_params': selected_params, 'selected_path': final.path_id,
        'note': 'One-SE uses variation of repeat means as a parsimony heuristic, not a confidence bound. Inner scores guide selection; outer CV evaluates the whole procedure.',
    }, indent=2, default=lambda v: v.item()), encoding='utf-8')
    metrics = pd.DataFrame(fold_rows)
    return {
        'model': fitted.pipeline_, 'support': fitted.support_, 'best_params': selected_params, 'selected_path': final.path_id,
        'cv_r2_mean': metrics.R2.mean(), 'cv_r2_std': metrics.R2.std(ddof=0),
        'cv_mae_mean': metrics.MAE.mean(), 'n_candidates': len(final.cv_results_['params']),
    }
