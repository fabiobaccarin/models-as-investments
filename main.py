"""
This file executes the exercise

Inputs: None
Outputs: results.csv
"""

import logging
import pandas as pd
import numpy as np
import models
import yaml
from sklearn import datasets, model_selection

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def _load_conf() -> dict:
    return yaml.safe_load(open("conf.yml"))

def _make_opts(conf: dict) -> dict:
    res = {
        arg: np.random.randint(**conf[arg])
        for arg in ["n_samples", "n_features"]
    }
    res["n_informative"] = (
        int(np.random.randint(1, 100)/100 * res["n_features"])
    )
    res["n_redundant"] = 0
    return res

def simulate_classification(conf) -> pd.DataFrame:
    """Performs simulation for classification"""
    def gen_data():
        return datasets.make_classification(
            flip_y=np.random.rand(),
            hypercube=False,
            **_make_opts(conf)
        )

    res = []
    logging.critical("Begin simulations")
    for d in range(conf["datasets"]):
        X = None
        while X is None:
            try:
                X, y = gen_data()
            except ValueError:
                X, y = gen_data()
        logging.critical(f"Simulating for dataset {d:02d}. Shape: {X.shape}")
        for name, model in models.CLASSIFICATION:
            out = model_selection.cross_validate(
                model, X, y,
                scoring=conf["scoring"]["classification"],
                cv=2,
                return_train_score=True,
                n_jobs=3,
            )
            res.append(
                {
                    "model": name,
                    "dataset": d,
                    "rows": len(X),
                    "cols": X.shape[1]
                }
                | {
                    f"test_{m}_avg": out[f"test_{m}"].mean()
                    for m in conf["scoring"]["classification"]
                }
            )
    return pd.DataFrame.from_records(res)

if __name__ == "__main__":
    conf = _load_conf()
    res = simulate_classification(conf)
    logging.critical("Saving results")
    res.to_csv(f"results/file{conf['file_id']}.csv")
    logging.critical("End of execution")