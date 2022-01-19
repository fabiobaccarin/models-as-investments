"""
This file executes the exercise

Inputs: None
Outputs: results.csv
"""

import logging
import uuid
import pandas as pd
import numpy as np
import models
import yaml
from sklearn import datasets, model_selection

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

FILE_ID = uuid.uuid4().hex

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

def simulate_classification(conf):
    """Performs simulation for classification"""
    def gen_data():
        while True:
            try:
                flip_y = np.random.rand()
                X, y = datasets.make_classification(
                    flip_y=flip_y,
                    hypercube=False,
                    **_make_opts(conf)
                )
                break
            except ValueError:
                continue
        return X, y, flip_y, uuid.uuid4().hex

    res = []
    logging.critical(f"Begin simulations: {conf['datasets']} datasets")
    for d in range(conf["datasets"]):
        X, y, flip_y, df_id = gen_data()
        logging.critical(f"Simulating for dataset {d:03d}. Shape: {X.shape}")
        for name, model in models.CLASSIFICATION:
            out = model_selection.cross_validate(
                model, X, y,
                scoring=conf["scoring"]["classification"],
                cv=2,
                return_train_score=True,
                n_jobs=-1,
            )
            res.append(
                {
                    "dataset": df_id,
                    "model": name,
                    "rows": len(X),
                    "cols": X.shape[1],
                    "flip_y": flip_y,
                }
                | {
                    f"tr_{m}": out[f"train_{m}"].mean()
                    for m in conf["scoring"]["classification"]
                }
                | {
                    f"te_{m}": out[f"test_{m}"].mean()
                    for m in conf["scoring"]["classification"]
                }
            )
    return pd.DataFrame.from_records(res)

if __name__ == "__main__":
    conf = _load_conf()
    res = simulate_classification(conf)
    logging.critical("Saving results")
    res.to_csv(f"results/{FILE_ID}.csv", index=False)
    logging.critical("End of execution")