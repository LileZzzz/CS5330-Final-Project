import mlflow

# Set the URI
mlflow.set_tracking_uri("sqlite:///scripts/mlflow.db")

# Find the experiment by name
experiment = mlflow.get_experiment_by_name("EuroSAT_ViT_Classification")

# Search the run
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'best_model'",
)

if len(runs) == 0:
    print("No run found")
else:
    # Get the run ID
    best_model_id = runs.iloc[0].run_id
    print(f"Found best_model with run ID: {best_model_id}")

    run = mlflow.get_run(best_model_id)

    # Get metrics
    metrics = run.data.metrics
    print("\n= METRICS =")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    print("\n= PARAMS =")
    # Get parameters
    params = run.data.params
    for param_name, param_value in params.items():
        print(f"{param_name}: {param_value}")
