import os
import logging
import importlib
import random

import torch
from typing import List, Optional, Tuple
from attention import Past
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import list_classes_from_module

logger = logging.getLogger(__name__)


class GPT2Handler(BaseHandler):

    corpus_path = os.environ.get("GPT2_CORPUS_PATH")
    nucleus_prob = 0.85
    seq_len = 64

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        """
        Loads the pickle file from the given model path.
        Args:
            model_dir (str): Points to the location of the model artefacts.
            model_file (.py): the file which contains the model class.
            model_pt_path (str): points to the location of the model pickle file.
        Raises:
            RuntimeError: It raises this error when the model.py file is missing.
            ValueError: Raises value error when there is more than one class in the label,
                        since the mapping supports only one label per class.
        Returns:
            serialized model file: Returns the pickled pytorch model file
        """
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError(
                "Expected only one class as model definition. {}".format(
                    model_class_definitions
                )
            )

        model_class = model_class_definitions[0]
        model = model_class().eval()
        if model_pt_path:
            state_dict = torch.load(model_pt_path, map_location=self.device)
            state_dict = state_dict["model"]
            model.load_state_dict(state_dict)
        return model

    def initialize(self, context):

        properties = context.system_properties
        self.map_location = (
            "cuda"
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            logger.debug("Loading eager model")
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
            self.model.to(self.device)
        else:
            logger.debug("Loading torchscript model")
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")

            self.model = self._load_torchscript_model(model_pt_path)

        # Doing this model evaluation breaks the results from inference
        # self.model.eval()

        logger.debug("Model file %s loaded successfully", model_pt_path)

        self.initialized = True

    def generate(self, context: str) -> str:
        words = self.model.encode_context(context)

        current, past = words, None
        while len(words) < self.seq_len:
            # Predict the next word token from the given context.
            probs, past = self._predict_probs(current, past)
            next_word = self._sample_from_top_p(probs)

            # Change the context to the predicted word.
            words.append(next_word)
            current = [next_word]

        return self.model.decode_tokens(words)

    @torch.no_grad()
    def _predict_probs(
        self, words: List[int], past: Optional[List[Past]] = None
    ) -> Tuple[torch.Tensor, List[Past]]:
        x = torch.tensor(words, dtype=torch.long)
        x = self.model.decorate_sequence(
            x, offset=past[0][0].size(-2) if past is not None else 0
        )

        # if self.config.use_gpu:
        #     logits, past = self.model(x.cuda(), past)
        #     logits = logits.cpu().float()
        # else:
        logits, past = self.model(x, past)

        return logits[-1, :].softmax(-1), past

    def _sample_from_top_p(self, probs: torch.Tensor) -> int:
        probs, indices = probs.sort(descending=True)

        mask = probs.cumsum(-1) > self.nucleus_prob
        mask[0] = False
        probs.masked_fill_(mask, 0)

        # Sample from filtered distribution.
        return indices[probs.multinomial(1)[0]].item()

    def _random_word(self) -> str:
        line_gen = (
            line.strip("\n") for line in open(self.corpus_path, "r+", encoding="utf-8")
        )
        return random.choice(list(line_gen))

    def preprocess(self, data):
        batch_data = []
        for entry in data:
            context = None
            # images python version not compatible with walrus :(
            body = entry.get("body")
            if body:
                context = body.get("context")
            if context is None:
                # would be good to decouple this down the road
                # could remove a related asset for random word file
                context = self._random_word()
            batch_data.append(context)
        return batch_data

    def inference(self, data, *args, **kwargs):
        inferences = []
        for context in data:
            inferences.append(self.generate(context))
        return inferences

    def postprocess(self, inference_output):
        results = []
        for result in inference_output:
            results.append(result.replace("<s>", "").replace("</s>", "").strip())
        return [{ "result": result } for result in results]
