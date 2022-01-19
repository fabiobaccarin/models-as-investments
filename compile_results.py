"""
This script reads all files in the results directory and produces a single
results data set
"""

def main():
    import pathlib
    import pandas as pd

    results = pathlib.Path(__file__).parent.joinpath("results")

    df = pd.concat([pd.read_csv(file) for file in results.iterdir()])

    df.to_pickle("results.pkl")

if __name__ == "__main__":
    main()