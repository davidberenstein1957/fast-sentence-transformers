# -*- coding: utf-8 -*-
import json
import logging
import os
from typing import Iterable, Optional, Union

import numpy as np
import onnxruntime
import psutil
import torch
import torch as t
from sentence_transformers import __MODEL_HUB_ORGANIZATION__, __version__
from sentence_transformers.models import Pooling
from sentence_transformers.util import snapshot_download
from torch import nn
from transformers import AutoTokenizer
from txtai.pipeline import HFOnnx

logger = logging.getLogger(__name__)


class FastSentenceTransformer(object):
    def __init__(
        self,
        model_name_or_path: str,
        modules: Optional[Iterable[nn.Module]] = None,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        use_auth_token: Union[bool, str, None] = None,
        onnx_folder: str = "models",
        enable_overwrite: bool = True,
        quantize: bool = False,
    ):
        self.quantize = quantize

        if cache_folder is None:
            cache_folder = os.getenv("SENTENCE_TRANSFORMERS_HOME")
            if cache_folder is None:
                try:
                    from torch.hub import _get_torch_home

                    torch_cache_home = _get_torch_home()
                except ImportError:
                    torch_cache_home = os.path.expanduser(
                        os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
                    )

                cache_folder = os.path.join(torch_cache_home, "sentence_transformers")

        if model_name_or_path is not None and model_name_or_path != "":
            logger.info("Load pretrained SentenceTransformer: {}".format(model_name_or_path))

            # Old models that don't belong to any organization
            basic_transformer_models = [
                "albert-base-v1",
                "albert-base-v2",
                "albert-large-v1",
                "albert-large-v2",
                "albert-xlarge-v1",
                "albert-xlarge-v2",
                "albert-xxlarge-v1",
                "albert-xxlarge-v2",
                "bert-base-cased-finetuned-mrpc",
                "bert-base-cased",
                "bert-base-chinese",
                "bert-base-german-cased",
                "bert-base-german-dbmdz-cased",
                "bert-base-german-dbmdz-uncased",
                "bert-base-multilingual-cased",
                "bert-base-multilingual-uncased",
                "bert-base-uncased",
                "bert-large-cased-whole-word-masking-finetuned-squad",
                "bert-large-cased-whole-word-masking",
                "bert-large-cased",
                "bert-large-uncased-whole-word-masking-finetuned-squad",
                "bert-large-uncased-whole-word-masking",
                "bert-large-uncased",
                "camembert-base",
                "ctrl",
                "distilbert-base-cased-distilled-squad",
                "distilbert-base-cased",
                "distilbert-base-german-cased",
                "distilbert-base-multilingual-cased",
                "distilbert-base-uncased-distilled-squad",
                "distilbert-base-uncased-finetuned-sst-2-english",
                "distilbert-base-uncased",
                "distilgpt2",
                "distilroberta-base",
                "gpt2-large",
                "gpt2-medium",
                "gpt2-xl",
                "gpt2",
                "openai-gpt",
                "roberta-base-openai-detector",
                "roberta-base",
                "roberta-large-mnli",
                "roberta-large-openai-detector",
                "roberta-large",
                "t5-11b",
                "t5-3b",
                "t5-base",
                "t5-large",
                "t5-small",
                "transfo-xl-wt103",
                "xlm-clm-ende-1024",
                "xlm-clm-enfr-1024",
                "xlm-mlm-100-1280",
                "xlm-mlm-17-1280",
                "xlm-mlm-en-2048",
                "xlm-mlm-ende-1024",
                "xlm-mlm-enfr-1024",
                "xlm-mlm-enro-1024",
                "xlm-mlm-tlm-xnli15-1024",
                "xlm-mlm-xnli15-1024",
                "xlm-roberta-base",
                "xlm-roberta-large-finetuned-conll02-dutch",
                "xlm-roberta-large-finetuned-conll02-spanish",
                "xlm-roberta-large-finetuned-conll03-english",
                "xlm-roberta-large-finetuned-conll03-german",
                "xlm-roberta-large",
                "xlnet-base-cased",
                "xlnet-large-cased",
            ]

            if os.path.exists(model_name_or_path):
                # Load from path
                model_path = model_name_or_path
            else:
                # Not a path, load from hub
                if "\\" in model_name_or_path or model_name_or_path.count("/") > 1:
                    raise ValueError("Path {} not found".format(model_name_or_path))

                if "/" not in model_name_or_path and model_name_or_path.lower() not in basic_transformer_models:
                    # A model from sentence-transformers
                    model_name_or_path = __MODEL_HUB_ORGANIZATION__ + "/" + model_name_or_path

                model_path = os.path.join(cache_folder, model_name_or_path.replace("/", "_"))

                # Download from hub with caching
                snapshot_download(
                    model_name_or_path,
                    cache_dir=cache_folder,
                    library_name="sentence-transformers",
                    library_version=__version__,
                    ignore_files=["flax_model.msgpack", "rust_model.ot", "tf_model.h5"],
                    use_auth_token=use_auth_token,
                )

        onnxproviders = onnxruntime.get_available_providers()

        if device == "cpu":
            fast_onnxprovider = "CPUExecutionProvider"
        else:
            if "CUDAExecutionProvider" not in onnxproviders:
                fast_onnxprovider = "CPUExecutionProvider"
            else:
                fast_onnxprovider = "CUDAExecutionProvider"

        os.makedirs(onnx_folder, exist_ok=True)
        self.onnx_folder = onnx_folder
        self.model_path = model_path
        self.fast_onnxprovider = fast_onnxprovider
        self.cache_folder = cache_folder
        self.enable_overwrite = enable_overwrite
        self.export_model_name = os.path.join(self.onnx_folder, f"{model_path}_quantized_{quantize}.onnx".lower())

        if os.path.exists(os.path.join(self.model_path, "modules.json")):  # Load as SentenceTransformer model
            self.sbertmodel2onnx()
            self.session = self._load_sbert_session()
            self.pooling_model = self._load_sbert_pooling()
        else:
            raise ValueError("AutoModels are not Implemented!")

    def sbertmodel2onnx(self):
        """
        It takes a HuggingFace model, converts it to ONNX, and saves it to a specified location.
        """
        self.onnx = HFOnnx()

        model_json_path = os.path.join(self.model_path, "modules.json")
        with open(model_json_path) as fIn:
            modules_config = json.load(fIn)
        tf_from_s_path = os.path.join(self.model_path, modules_config[0].get("path"))

        tokenizer = AutoTokenizer.from_pretrained(tf_from_s_path, do_lower_case=True, cache_dir=self.cache_folder)
        self.tokenizer = tokenizer

        if os.path.exists(self.export_model_name):
            print(f"Model found at: {self.export_model_name}")
        else:
            self.onnx(self.model_path, "default", self.export_model_name, quantize=self.quantize)
            print(f"Model exported at: {self.export_model_name}")

    def encode(self, sentences: list) -> np.array:
        """
        1. The function takes a list of sentences as input.
        2. It then converts the sentences into a format that can be understood by the ONNX model.
        3. It then runs the ONNX model on the sentences and returns the embeddings

        :param sentences: list of strings
        :type sentences: list
        :return: The sentence embedding
        """
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]

        inputs = self.tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
        ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
        ort_outputs_gpu = self.session.run(None, ort_inputs)
        ort_result = self.pooling_model.forward(
            features={
                "token_embeddings": t.Tensor(ort_outputs_gpu[0]),
                "attention_mask": inputs.get("attention_mask"),
            }
        )
        result = ort_result.get("sentence_embedding")
        return result.cpu().detach().numpy()

    def _load_sbert_pooling(self):
        """
        It loads the pooling model from the path specified in the modules.json file.
        :return: The pooling model
        """
        model_json_path = os.path.join(self.model_path, "modules.json")
        with open(model_json_path) as fIn:
            modules_config = json.load(fIn)

        pooling_model_path = os.path.join(self.model_path, modules_config[1].get("path"))
        pooling_model = Pooling.load(pooling_model_path)
        return pooling_model

    def _load_sbert_session(self):
        """
        The function loads the ONNX model into an ONNXRuntime session, and sets the number of threads to the number of
        logical cores on the machine
        :return: The session is being returned.
        """
        sess_options = onnxruntime.SessionOptions()

        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)

        session = onnxruntime.InferenceSession(
            self.export_model_name, sess_options, providers=[self.fast_onnxprovider]
        )
        return session
