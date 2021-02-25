#!/usr/bin/env python3
from argparse import ArgumentParser
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
import tqdm

from janus.repair.mutate import get_random_mutation_sampler
from janus.pipeline import pipeline_to_tree as pt


class PipelineScorePredictor(object):
    def __init__(self, model=None, encoder=None):
        if model is None:
            model = RandomForestRegressor()
        self.model = model

        if encoder is None:
            encoder = CountVectorizer(
                tokenizer=pt.pipeline_tokenizer,
                token_pattern=None,
            )
        self.encoder = encoder

        self.fit_ = False

    def encode_(self, pipelines):
        tr = [pt.to_tree(p) for p in pipelines]
        jsons = [pt.to_json(t) for t in tr]
        as_text = [str(j) for j in jsons]

        # encode into text
        if not self.fit_:
            self.encoder.fit(as_text)
        enc = self.encoder.transform(as_text)
        return enc

    def fit(self, pipelines, scores):
        # convert to vector rep
        X = self.encode_(pipelines)
        y = np.array(scores)
        self.model.fit(X, y)
        self.fit_ = True
        return self

    def predict(self, pipelines):
        # convert to vector rep
        encoded = self.encode_(pipelines)
        return self.model.predict(encoded)


def train(paths, random_state=None):
    dfs = []
    for p in tqdm.tqdm(paths):
        df = pd.read_pickle(p)
        df["failed"] = df["external_score"].isna()
        df = df[~df["failed"]]
        df = df[["obj_graph", "external_score"]]
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0)
    pipelines = combined_df["obj_graph"].values
    scores = combined_df["external_score"].values

    if random_state is not None:
        np.random.seed(random_state)
    score_predictor = PipelineScorePredictor()
    score_predictor.fit(pipelines, scores)
    return score_predictor


def get_args():
    parser = ArgumentParser(
        description="Train meta learner pipeline score strategy")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="Path to pickled search trace dataframes",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to dump trained model",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="RNG seed",
        default=42,
    )
    return parser.parse_args()


def main():
    args = get_args()
    score_predictor = train(args.input)
    with open(args.output, "wb") as fout:
        pickle.dump(score_predictor, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
        sys.exit(1)
