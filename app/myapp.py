import os
import sys
import argparse
from typing import Optional
from typing import Any
from typing import Dict
from allennlp.commands.train import train_model_from_args
from allennlp.commands.train import train_model_from_file
from allennlp.commands.evaluate import evaluate_from_args
from allennlp.common import logging as common_logging
from allennlp.predictors import Predictor
from allennlp.commands.predict import _predict
from allennlp.commands.predict import _get_predictor
from allennlp.commands.predict import _PredictManager

# These imports are important for making the configuration files find the classes that you wrote.
# If you don't have these, you'll get errors about allennlp not being able to find
# "simple_classifier", or whatever name you registered your model with.  These imports and the
# contents of .allennlp_plugins makes it so you can just use `allennlp train`, and we will find your
# classes and use them.  If you change the name of `my_project`, you'll also need to change it in
# the same way in the .allennlp_plugins file.
# from my_project.model import *
# from my_project.dataset_reader import *

class Application:

    def train(self) -> Optional[Any]:
        print(os.getcwd())
        allennlp_args = argparse.Namespace()
        allennlp_args.param_path = "training_config/my_model_trained_on_my_dataset.jsonnet"
        allennlp_args.serialization_dir = "result"
        allennlp_args.overrides = ""
        allennlp_args.recover = False
        allennlp_args.force = False
        allennlp_args.node_rank = 0
        # allennlp_args.include_package = ["project_test"]
        allennlp_args.include_package = None
        allennlp_args.dry_run = False
        allennlp_args.file_friendly_logging = False

        print(allennlp_args)
        print("")

        model = train_model_from_args(allennlp_args)
        return model

    def train_file(self):
        config_filename = "training_config/my_model_trained_on_my_dataset.jsonnet"
        serialization_dir = "result"
        train_model_from_file(
            config_filename, serialization_dir, file_friendly_logging=True, force=True
        )

    def restore_and_evaluate(self) -> Dict[str, Any]:
        allennlp_args = argparse.Namespace()
        allennlp_args.file_friendly_logging = False
        allennlp_args.archive_file = "result/model.tar.gz"
        allennlp_args.weights_file = None
        allennlp_args.cuda_device = -1
        allennlp_args.overrides = ""
        allennlp_args.input_file = "data/movie_review/test.tsv"
        allennlp_args.embedding_sources_mapping = ""
        allennlp_args.extend_vocab = False
        allennlp_args.batch_size = None
        allennlp_args.batch_weight_key = None
        allennlp_args.output_file = "evaluation/evaluation"
        allennlp_args.predictions_output_file = "evaluation/pred"

        metric = evaluate_from_args(allennlp_args)

    def restore_and_predict(self):
        allennlp_args = argparse.Namespace()
        allennlp_args.archive_file = "result/model.tar.gz"
        allennlp_args.input_file = "data/movie_review/test.jsonl"
        allennlp_args.output_file = "predct"
        allennlp_args.weights_file = None
        allennlp_args.batch_size = 1
        allennlp_args.silent = False
        allennlp_args.cuda_device = -1
        allennlp_args.use_dataset_reader = None
        allennlp_args.dataset_reader_choice = "validation"
        allennlp_args.overrides = ""
        allennlp_args.predictor = "project_test.predictor.SentenceClassifierPredictor"
        allennlp_args.predictor_args = ""
        allennlp_args.file_friendly_logging = False

        _predict(allennlp_args)

    def construct_params_for_predict(self) -> argparse.Namespace:
        allennlp_args = argparse.Namespace()
        allennlp_args.archive_file = "result/model.tar.gz"
        allennlp_args.input_file = "data/movie_review/test.jsonl"
        allennlp_args.output_file = "predct"
        allennlp_args.weights_file = None
        allennlp_args.batch_size = 1
        allennlp_args.silent = False
        allennlp_args.cuda_device = -1
        allennlp_args.use_dataset_reader = None
        allennlp_args.dataset_reader_choice = "validation"
        allennlp_args.overrides = ""
        allennlp_args.predictor = "project_test.predictor.SentenceClassifierPredictor"
        allennlp_args.predictor_args = ""
        allennlp_args.file_friendly_logging = False
        return allennlp_args

    def restore_predictor(self, args: argparse.Namespace) -> Predictor:
        common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging

        predictor = _get_predictor(args)

        if args.silent and not args.output_file:
            print("--silent specified without --output-file.")
            print("Exiting early because no output will be created.")
            sys.exit(0)

        return predictor

    def predict(self, args: argparse.Namespace, predictor: Predictor):
        manager = _PredictManager(
            predictor,
            args.input_file,
            args.output_file,
            args.batch_size,
            not args.silent,
            args.use_dataset_reader,
        )
        manager.run()

    def predict_json(self, input_json: Dict[str, Any], predictor:Predictor):
        input_json = input_json if input_json else {"sentence": "A good movie!"}
        output = predictor.predict_json(input_json)
        return output

    def run_with_console_out(self):
        self.train()

    def run_with_file_out(self):
        stdout_check_point = sys.stdout
        stderr_check_point = sys.stderr
        with open("out.txt", "w+") as fw:
            sys.stdout = fw
            sys.stderr = fw
            self.train()
            print("\n\n[End of Logging]\n\n")
        sys.stdout = stdout_check_point
        sys.stderr = stderr_check_point


if __name__ == "__main__":
    app = Application()
    args = app.construct_params_for_predict()
    predictor = app.restore_predictor(args)
    result = app.predict_json({"sentence": "A good movie!"}, predictor)
    print(result)
    # restore_and_predict()
