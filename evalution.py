import os
import json
import yaml
import argparse
from pathlib import Path
from anomalib.engine import Engine
from train import load_config, build_datamodule, build_model

os.environ["TRUST_REMOTE_CODE"] = "1"

def main():
    parser = argparse.ArgumentParser(description="Build patchcore on dataset")
    parser.add_argument("--config",
                        type = str,
                        default= "configs/resnet18_config.yaml",
                        help= "Pass config path to run test on data")
    args = parser.parse_args()
    config = load_config(args.config)
    model = build_model(config["model"])
    datamodule = build_datamodule(config["data"])
    engine = Engine()
    results = engine.test(model=model, datamodule=datamodule)
    metrics = results[0]

    print("\n Evaluation results")

    for k, v in metrics.items():
        print(f"{k} : {v:.4f}")

if __name__=="__main__":
    main()