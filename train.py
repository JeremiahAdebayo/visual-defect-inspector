import os
import torch
import mlflow
import argparse
import yaml
import json
import datetime
from pathlib import Path
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import MVTecAD

os.environ["TRUST_REMOTE_CODE"] = "1"

save_dir = Path("./saved_model")
save_dir.mkdir(parents=True, exist_ok=True)

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def prepare_data(cfg: dict):
    print(f"Loading dataset, category = {cfg['category']}, root = {cfg['root']}")
    datamodule = MVTecAD(root = cfg['root'],
                         category = cfg['category'],
                         train_batch_size = cfg['train_batch_size'],
                         eval_batch_size = cfg['eval_batch_size'],
                         num_workers = cfg['num_workers'])
    
    datamodule.prepare_data()
    datamodule.setup()

    return datamodule

def build_model(cfg: dict):
    print(f"Building model with backbone {cfg['backbone']}")
    model =  Patchcore(backbone = cfg['backbone'],
                       layers = cfg['layers'],
                       num_neigbors = cfg['num_neighbors'],
                       coreset_sampling_ratio = cfg['coreset_sampling_ratio'])
    return model 

def train(config: dict, config_path: str) -> None:
    os.environ["TRUST_REMOTE_CODE"] = "1"

    save_dir = Path(config["artifacts"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    datamodule = prepare_data(config["data"])
    model = build_model(config["model"])

    engine = Engine(
        max_epochs=config["engine"]["max_epochs"],
        enable_progress_bar=config["engine"]["enable_progress_bar"],
        enable_model_summary=config["engine"]["enable_model_summary"],
        logger=config["engine"]["logger"],
    )

    print("Training...")
    engine.fit(model=model, datamodule=datamodule)

    print("Evaluating...")
    results = engine.test(model=model, datamodule=datamodule)
    metrics = results[0]

    print("\n----- Results -----")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # MLflow
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    run_name = f"patchcore-{config['data']['category']}-{config['model']['backbone']}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("backbone", config["model"]["backbone"])
        mlflow.log_param("layers", config["model"]["layers"])
        mlflow.log_param("coreset_sampling_ratio", config["model"]["coreset_sampling_ratio"])
        mlflow.log_param("num_neighbors", config["model"]["num_neighbors"])
        mlflow.log_param("category", config["data"]["category"])
        mlflow.log_param("image_size", config["data"]["image_size"])
        mlflow.log_artifact(config_path)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    # Save artifacts
    engine.export(model=model, export_type="torch", export_root=save_dir)
    torch.save(model.model.memory_bank, save_dir / "memory_bank.pt")

    metadata = {
        "backbone": config["model"]["backbone"],
        "layers": config["model"]["layers"],
        "category": config["data"]["category"],
        "coreset_sampling_ratio": config["model"]["coreset_sampling_ratio"],
        "num_neighbors": config["model"]["num_neighbors"],
        "image_size": config["data"]["image_size"],
        "metrics": {k: float(v) for k, v in metrics.items()},
        "trained_at": datetime.now().isoformat(),
    }

    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts saved to {save_dir}")
    print(f"Memory bank size: {model.model.memory_bank.shape}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PatchCore on MVTecAD")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet18_config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    train(config, args.config)