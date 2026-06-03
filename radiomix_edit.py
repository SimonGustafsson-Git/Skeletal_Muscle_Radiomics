import os, sys, json, timeit, datetime, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import joblib
from joblib import parallel_backend
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import GridSearchCV, ParameterGrid, RepeatedStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import radiomix_pipeline_classes as rpc
import radMLBench
import utils


def adjust_param_grid(param_grid, ovr_type, clf_name):
    """Adjust parameter grid based on whether OvR is used due to sklearn's OVR requirements"""
    if ovr_type == 'With OvR':
        adjusted_grid = {k.replace('classifier__', 'classifier__estimator__'): v
                         for k, v in param_grid.items()}

        if clf_name == 'LR' and 'classifier__estimator__multi_class' not in adjusted_grid:
            adjusted_grid['classifier__estimator__multi_class'] = ['ovr']
    else:
        adjusted_grid = {k.replace('estimator__', ''): v for k, v in param_grid.items()}
    return adjusted_grid


def update_results_summary(summary_result, main_dir):
    """
    Updates the results summary table with the latest dataset results

    Parameters:
    -----------
    summary_result : dict
        Dictionary containing the summary results for a dataset
    main_dir : Path
        Path to the main directory
    """
    summary_file = main_dir / "RadiomiX_results_summary.csv"

    current_result_df = pd.DataFrame([summary_result])

    if summary_file.exists():
        try:
            summary_df = pd.read_csv(summary_file)
            # Keep compatibility with older summary files by adding any new columns.
            for col in current_result_df.columns:
                if col not in summary_df.columns:
                    summary_df[col] = np.nan
            current_result_df = current_result_df.reindex(columns=summary_df.columns, fill_value=np.nan)

            # Check if this dataset is already in the summary, if not, append it
            if summary_result['dataset_name'] in summary_df['dataset_name'].values:
                existing_idx = summary_df[summary_df['dataset_name'] == summary_result['dataset_name']].index
                existing_score = summary_df.loc[existing_idx, 'best_score'].values[0]

                if summary_result['best_score'] > existing_score:
                    summary_df.loc[existing_idx] = current_result_df.iloc[0]
            else:
                summary_df = pd.concat([summary_df, current_result_df], ignore_index=True)

        except Exception as e:
            print(f"Error updating summary file: {e}")
            print("Creating a new summary file")
            summary_df = current_result_df
    else:
        summary_df = current_result_df

    # Sorting
    summary_df = summary_df.sort_values(by='best_score', ascending=False)
    summary_df.to_csv(summary_file, index=False)
    # Also create a formatted HTML report for better visualization
    html_file = main_dir / "RadiomiX_results_summary.html"

    # Format the summary table for HTML display
    styled_df = summary_df.style.highlight_max(subset=['best_score'], color='lightgreen') \
        .format({'best_score': '{:.4f}'}) \
        .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f2f2f2'),
                                     ('color', 'black'),
                                     ('font-weight', 'bold')]},
        {'selector': 'td', 'props': [('padding', '5px')]}
    ])

    # Save the HTML report
    with open(html_file, 'w') as f:
        f.write("<html><head><title>RadiomiX AutoML Results Summary</title>")
        f.write("<style>body{font-family:Arial,sans-serif;margin:20px;}</style>")
        f.write("</head><body>")
        f.write("<h1>RadiomiX AutoML Results Summary</h1>")
        f.write(f"<p>Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        f.write(styled_df.to_html())
        f.write("</body></html>")

    print(f"\nResults summary updated and saved to:")
    print(f"  - CSV: {summary_file}")
    print(f"  - HTML: {html_file}")


def summarize_all_results(main_dir=None):
    """
    Generate a comprehensive summary of results across all datasets

    Parameters:
    -----------
    main_dir : Path or str, optional
        Path to the main directory. If None, uses current working directory.

    Returns:
    --------
    DataFrame
        Summary DataFrame with results
    """

    if main_dir is None:
        main_dir = Path(os.getcwd())
    else:
        main_dir = Path(main_dir)

    summary_file = main_dir / "RadiomiX_results_summary.csv"

    if not summary_file.exists():
        print("No summary file found. Run the pipeline on at least one dataset first.")
        return pd.DataFrame()

    # Load the summary data
    summary_df = pd.read_csv(summary_file)

    print("\n=== RADIOMIX AUTOML RESULTS SUMMARY ===")
    print(f"Number of datasets analyzed: {len(summary_df)}")

    if len(summary_df) > 0:
        best_dataset = summary_df.loc[summary_df['best_score'].idxmax()]
        print(f"\nBest overall result:")
        print(f"  Dataset: {best_dataset['dataset_name']}")
        print(f"  Score: {best_dataset['best_score']:.4f}")
        print(f"  Classifier: {best_dataset['best_classifier']}")
        print(f"  Feature Selector: {best_dataset['best_feature_selector']}")
        print(f"  OvR: {best_dataset['best_ovr']}")
        if 'best_hyperparameters' in summary_df.columns:
            print(f"  Best Hyperparameters: {best_dataset['best_hyperparameters']}")

        # Create a simplified view for display
        display_cols = ['dataset_name', 'best_classifier', 'best_feature_selector', 'best_ovr', 'best_score']
        if 'best_hyperparameters' in summary_df.columns:
            display_cols.append('best_hyperparameters')
        display_df = summary_df[display_cols].sort_values(by='best_score', ascending=False)

        print("\nAll Results:")
        print(display_df.to_string(index=False))

        # Generate additional insights
        print("\nInsights:")
        clf_counts = summary_df['best_classifier'].value_counts()
        fs_counts = summary_df['best_feature_selector'].value_counts()

        print(f"Most successful classifier: {clf_counts.index[0]} (used in {clf_counts.iloc[0]} best models)")
        print(f"Most successful feature selector: {fs_counts.index[0]} (used in {fs_counts.iloc[0]} best models)")

    return summary_df


def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar"""

    class TqdmBatchCompletionCallback:
        def __init__(self, tqdm_object):
            self.tqdm_object = tqdm_object

        def __call__(self, *args, **kwargs):
            self.tqdm_object.update(n=1)
            return None

    class TqdmJoblib:
        def __init__(self, tqdm_obj):
            self.tqdm_obj = tqdm_obj

        def __enter__(self):
            self.old_batch_callback = joblib.parallel.BatchCompletionCallBack
            joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback(self.tqdm_obj)
            return self.tqdm_obj

        def __exit__(self, exc_type, exc_val, exc_tb):
            joblib.parallel.BatchCompletionCallBack = self.old_batch_callback
            self.tqdm_obj.close()

    return TqdmJoblib(tqdm_object)


def run_radiomix_automl(
        dataset_name=None,
        dataset_path=None,
        param_file="full_params",
        target_column="Target",
        id_column="ID",
        use_ovr=False,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        key_metric="auc"
):
    """
    Main function to run the RadiomiX AutoML pipeline

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    dataset_path : str
        Path to the dataset if not using radMLBench
    param_file : str
        Which parameter file to use: "full_params", "fast_params.json", or "custom"
    target_column : str
        Name of the target column
    id_column : str
        Name of the ID column to drop
    random_state : int
        Random state for reproducibility
    n_jobs : int
        Number of jobs for parallel processing
    verbose : int
        Verbosity level
    """

    print(f"Starting RadiomiX AutoML with {param_file} parameters")

    # Load the dataset
    if dataset_path:
        # Load from provided path
        print(f"Loading dataset from path: {dataset_path}")
        data = pd.read_csv(dataset_path)
    elif dataset_name:
        # Load from radMLBench
        print(f"Loading dataset from radMLBench: {dataset_name}")
        data = radMLBench.loadData(dataset_name)
    else:
        raise ValueError("Either dataset_name or dataset_path must be provided")

    # Create a directory for the dataset if it doesn't exist
    main_dir = Path(os.getcwd())
    radiomix_dir = Path(__file__).resolve().parent
    output_base_dir = radiomix_dir / "Output_data"
    dataset_name = dataset_name or Path(dataset_path).stem
    results_dir = output_base_dir / f"output_{dataset_name}"
    results_dir.mkdir(parents=True, exist_ok=True)
    pipeline_cache = joblib.Memory(location=str(results_dir / "_pipeline_cache"), verbose=0)

    print(f"Results will be saved to: {results_dir}")
    print(f"Dataset shape: {data.shape}")

    # Separate target column and drop ID column
    if target_column in data.columns:
        y = data[target_column]
        X = data.drop(target_column, axis=1)
        print(f"Target column '{target_column}' found with value counts:")
        print(y.value_counts())
        # Handle missing values if any
        if y.isnull().values.any():
            raise ValueError(f"Target column '{target_column}' has missing values")

    else:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    if id_column in X.columns:
        X = X.drop(id_column, axis=1)
        print(f"Dropped ID column '{id_column}'")

    # Define scoring metrics based on number of classes
    if y.nunique() > 2:
        scoring = {"auc": "roc_auc_ovr", "accuracy": "accuracy", "f1": "f1_macro"}
    else:
        scoring = {"auc": "roc_auc", "accuracy": "accuracy", "f1": "f1"}

    #### Oversampling ####
    oversampler_options = ['passthrough']

    #### FS and CLF param grid ####
    # Load parameter sets based on param_file argument
    param_file_path = main_dir / f"{param_file}.json"
    if not param_file_path.exists():
        # Create default parameter files if they don't exist
        utils.create_default_param_files(main_dir)

    with open(param_file_path, 'r') as f:
        params = json.load(f)
        print(f"{param_file} loaded successfully")
    selector_configs = params.get('selector_configs', {})
    classifier_params = params.get('classifiers', {})

    classifier_instances = {
        'RF': RandomForestClassifier(random_state=random_state),
        'LR': LogisticRegression(random_state=random_state),
        'ADABoost': AdaBoostClassifier(random_state=random_state),
        'XGBoost': XGBClassifier(random_state=random_state),
        'SVM': SVC(random_state=random_state, probability=True),
        'KNN': KNeighborsClassifier(),
        'Bayes': MultinomialNB()
    }

    classifiers = {}
    for clf_name in classifier_params:
        if clf_name in classifier_instances:
            param_grid = classifier_params[clf_name].copy()
            param_grid['oversampler'] = oversampler_options
            classifiers[clf_name] = (classifier_instances[clf_name], param_grid)

    #### Multiple train-test split strategy ####
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=random_state)

    results = []
    pipeline_starting_time = timeit.default_timer()
    best_params = []
    # Start at -inf so we always keep the actual best model/parameter combination found.
    best_score = float("-inf")
    best_pipeline = None
    critical_save = None

    #### Run the pipeline and GridSearch ####
    for clf_name, (clf, param_grid) in classifiers.items():
        for feature_selector_name, methods in selector_configs.items():
            print(f"Running pipeline with {clf_name} classifier and {feature_selector_name} feature selector")
            feature_selector = rpc.FeatureSelector(methods=methods, verbose=verbose)

            # Create pipelines based on use_ovr flag
            if use_ovr:
                pipelines = {
                    'With OvR': Pipeline([
                        ('feature_selector', feature_selector),
                        ('scaler', rpc.CustomScaler(clf_name, scaler_type="Standard", verbose=verbose)),
                        ('oversampler', 'passthrough'),
                        ('classifier', OneVsRestClassifier(estimator=clf))
                    ], memory=pipeline_cache)
                }
            else:
                pipelines = {
                    'Without OvR': Pipeline([
                        ('feature_selector', feature_selector),
                        ('scaler', rpc.CustomScaler(clf_name, scaler_type="Standard", verbose=verbose)),
                        ('oversampler', 'passthrough'),
                        ('classifier', clf)
                    ], memory=pipeline_cache)
                }

            for ovr_type, pipeline in pipelines.items():
                # Adjust parameter grid based on OvR classifier since sklearn OVR has a different naming rules
                adjusted_param_grid = adjust_param_grid(param_grid, ovr_type, clf_name)

                # Compute total tasks for progress bar
                n_candidates = len(list(ParameterGrid(adjusted_param_grid)))
                n_splits = cv.n_splits if hasattr(cv, 'n_splits') else cv.get_n_splits(X, y)
                total_tasks = n_candidates * n_splits

                # Run GridSearchCV with progress bar
                with parallel_backend('loky', n_jobs=n_jobs):
                    grid_search = GridSearchCV(
                        pipeline,
                        adjusted_param_grid,
                        cv=cv,
                        scoring=scoring,
                        error_score='raise',
                        verbose=1,
                        refit=key_metric
                    )
                    grid_search.fit(X, y)

                print("GridSearchCV completed!")
                print("Best AUC:", grid_search.cv_results_["mean_test_auc"][grid_search.best_index_])
                print("Best Accuracy:", grid_search.cv_results_["mean_test_accuracy"][grid_search.best_index_])

                #### Calculate results ####
                result_dict = rpc.metrics_calculator(
                    grid_search,
                    clf_name,
                    ovr_type,
                    feature_selector_name
                )
                result_dict["classifier"] = clf_name
                result_dict["feature_selector"] = feature_selector_name
                result_dict["ovr_type"] = ovr_type

                # Set a df aside with the results and a df for results that overcome the performance threshold
                results.append(result_dict)
                results_df = pd.DataFrame([result_dict])

                if result_dict[key_metric+"_best"] > best_score:
                    best_params = [clf_name, grid_search.best_params_, ovr_type, feature_selector_name]
                    best_score = result_dict[key_metric+"_best"]
                    best_pipeline = grid_search.best_estimator_
                    print(f"Best {key_metric} so far is {best_score}")
                    critical_save = pd.DataFrame(grid_search.cv_results_)

    #### Save results once after all classifiers ####
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / "Results_Summary" / target_column
    results_path.mkdir(parents=True, exist_ok=True)
    results_file_path = results_path / "Results.csv"
    pd.DataFrame(results).to_csv(results_file_path, index=False)

    # Save critical info for the global best result
    if critical_save is not None:
        critical_save.to_csv(results_path / "best_cv_results.csv")

    pipeline_stopping_time = timeit.default_timer()
    print(f'Total time for {clf_name}: {round((pipeline_stopping_time - pipeline_starting_time) / 60, 3)} Minutes')
    pipeline_starting_time = timeit.default_timer()

    # Save best model and parameters
    print("\nBest configuration:")
    print(f"Classifier: {best_params[0]}")
    print(f"Parameters: {best_params[1]}")
    print(f"OvR: {best_params[2]}")
    print(f"Feature Selector: {best_params[3]}")
    print(f"Best Score ({key_metric}): {best_score}")

    # Save best pipeline
    if best_pipeline:
        joblib.dump(best_pipeline, results_dir / "best_pipeline.joblib")

    # For better backup - Save all results in a single DataFrame
    all_results_df = pd.DataFrame(results)
    all_results_df.to_csv(results_dir / "all_results.csv", index=False)
    # Save best result per classifier
    per_model_df = pd.DataFrame(results)

    # Keep the best row for each classifier based on AUC
    best_per_model_df = (
        per_model_df.sort_values("auc_best", ascending=False)
        .drop_duplicates(subset=["classifier"])
        .reset_index(drop=True)
    )
    best_per_model_df.to_csv(results_dir / "best_per_model.csv", index=False)

    #### Create results summary ####
    summary_result = {
        'dataset_name': dataset_name,
        'timestamp': timestamp,
        'target_column': target_column,
        'best_classifier': best_params[0] if best_params else None,
        'best_hyperparameters': json.dumps(best_params[1], sort_keys=True) if best_params else None,
        'best_feature_selector': best_params[3] if best_params else None,
        'best_ovr': best_params[2] if best_params else None,
        'best_score': best_score,
        'num_samples': len(X),
        'num_features': X.shape[1],
        'class_distribution': dict(y.value_counts()),
        'param_file': param_file
    }
    # Save and update summary inside this dataset's output folder
    update_results_summary(summary_result, results_dir)
    return best_pipeline, best_params, best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RadiomiX AutoML')
    parser.add_argument('--dataset', type=str, help='Dataset name from radMLBench')
    parser.add_argument('--path', type=str, help='Path to dataset CSV file')
    parser.add_argument('--params', type=str, default='fast_params',
                        choices=['full_params', 'fast_params', 'custom_params'],
                        help='Parameter set to use')
    parser.add_argument('--target', type=str, default='Target', help='Target column name')
    parser.add_argument('--id', type=str, default='ID', help='ID column name')
    parser.add_argument('--use_ovr', default=False, action='store_true', help='Use One-vs-Rest classification')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of jobs for parallel processing')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level')
    parser.add_argument('--summary', action='store_true', help='Only show the summary of all results')

    args = parser.parse_args()

    radiomix_dir = Path(__file__).resolve().parent
    output_base_dir = radiomix_dir / "Output_data"
    resolved_dataset_name = args.dataset or (Path(args.path).stem if args.path else None)
    resolved_results_dir = (output_base_dir / f"output_{resolved_dataset_name}") if resolved_dataset_name else None

    # If summary flag is provided, only show the summary without running the pipeline
    if args.summary:
        if resolved_results_dir is None:
            print("Please provide --dataset or --path together with --summary.")
            sys.exit(1)
        summarize_all_results(resolved_results_dir)
        sys.exit(0)

    if not args.dataset and not args.path:
        print("Available datasets from radMLBench:")
        for dataset in radMLBench.listDatasets():
            print(f"  - {dataset}")
        print("\nPlease specify either --dataset or --path")
        sys.exit(1)

    run_radiomix_automl(
        dataset_name=args.dataset,
        dataset_path=args.path,
        param_file=args.params,
        target_column=args.target,
        id_column=args.id,
        use_ovr=args.use_ovr,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=args.verbose
    )

    # Show updated summary after running
    if resolved_results_dir is not None:
        summarize_all_results(resolved_results_dir)
