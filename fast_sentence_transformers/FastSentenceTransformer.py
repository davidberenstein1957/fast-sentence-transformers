# -*- coding: utf-8 -*-
import json
import logging
import os
from typing import Iterable, List, Optional, Union

import numpy as np
import onnxruntime
import psutil
import torch
import torch as t
from sentence_transformers import __MODEL_HUB_ORGANIZATION__, __version__
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.util import snapshot_download
from torch import Tensor, nn
from tqdm.autonotebook import trange
from transformers import AutoTokenizer

from fast_sentence_transformers.txtai import HFOnnx

logger = logging.getLogger(__name__)


class FastSentenceTransformer(object):
    def __init__(
        self,
        model_name_or_path: str,
        modules: Optional[Iterable[nn.Module]] = None,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        use_auth_token: Union[bool, str, None] = None,
        enable_overwrite: bool = True,
        quantize: bool = False,
    ):
        self.device = device
        self.quantize = quantize

        if cache_folder is None:
            cache_folder = os.getenv("SENTENCE_TRANSFORMERS_HOME")
            if cache_folder is None:
                try:
                    from torch.hub import _get_torch_home

                    torch_cache_home = _get_torch_home()
                except ImportError:
                    torch_cache_home = os.path.expanduser(
                        os.getenv(
                            "TORCH_HOME",
                            os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"),
                        )
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

                if not os.path.exists(os.path.join(model_path, "modules.json")):
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
                logger.warning("Using CPU. Try installing 'onnxruntime-gpu'.")
                fast_onnxprovider = "CPUExecutionProvider"
            else:
                fast_onnxprovider = "CUDAExecutionProvider"

        self.model_path = model_path
        self.fast_onnxprovider = fast_onnxprovider
        self.cache_folder = cache_folder
        self.enable_overwrite = enable_overwrite

        if os.path.exists(os.path.join(self.model_path, "modules.json")):  # Load as SentenceTransformer model
            self._load_sbert_model()
        else:  # Load with AutoModel
            self._load_auto_model()
        self.model2onnx()
        self._load_session()

    def model2onnx(self):
        """
        It takes a HuggingFace model, converts it to ONNX, and saves it to a specified location.
        """
        self.onnx = HFOnnx()

        if os.path.exists(os.path.join(self.model_path, "modules.json")):
            model_json_path = os.path.join(self.model_path, "modules.json")
            with open(model_json_path) as fIn:
                modules_config = json.load(fIn)
            tf_from_s_path = os.path.join(self.model_path, modules_config[0].get("path"))
            tokenizer = AutoTokenizer.from_pretrained(
                tf_from_s_path, do_lower_case=True, cache_dir=self.cache_folder, fast=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, do_lower_case=True, cache_dir=self.cache_folder, fast=True
            )
        self.tokenizer = tokenizer

        self.export_model_name = os.path.join(self.model_path, f"quantized_{self.quantize}.onnx".lower())

        if os.path.exists(self.export_model_name):
            print(f"Model found at: {self.export_model_name}")
        else:
            self.onnx(self.model_path, "default", self.export_model_name, quantize=self.quantize)
            print(f"Model exported at: {self.export_model_name}")

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings.
            Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors.
            Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return.
            Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1.
            In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned.
            If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        if device:
            logger.warning(
                f"Device can only be set during model FastSentenceTransformer initialization. Using {self.device}."
            )

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]

            if output_value == "token_embeddings":
                embeddings = self.encode_batch(sentences_batch, "token_embeddings")
            elif output_value is None:  # Return all outputs
                embeddings = self.encode_batch(sentences_batch, None)
            elif output_value == "sentence_embedding":  # Sentence embeddings
                embeddings = self.encode_batch(sentences_batch, "sentence_embedding")
                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu().detach().numpy()

                if normalize_embeddings:
                    norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
                    embeddings = embeddings / np.where(norms < 1e-12, 1e-12, norms)
            else:
                raise NotImplementedError(
                    "`output_value` can only be `token_embeddings`, `sentence_embedding` or None"
                )
            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def encode_batch(self, sentences: list, output_value: str) -> np.array:
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
        if output_value == "token_embeddings":
            embeddings = []
            for token_emb, attention in zip(ort_result.get("token_embeddings"), ort_result.get("attention_mask")):
                last_mask_id = len(attention) - 1
                while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                    last_mask_id -= 1

                embeddings.append(token_emb[0 : last_mask_id + 1])
        elif output_value is None:
            embeddings = []
            for sent_idx in range(len(ort_result.get("sentence_embedding"))):
                row = {name: ort_result[name][sent_idx] for name in ort_result}
                embeddings.append(row)
        else:
            embeddings = ort_result.get("sentence_embedding")
        return embeddings

    def _load_sbert_model(self):
        """
        Loads a full sentence-transformers model
        """
        # Check if the config_sentence_transformers.json file exists (exists since v2 of the framework)
        config_sentence_transformers_json_path = os.path.join(self.model_path, "config_sentence_transformers.json")
        if os.path.exists(config_sentence_transformers_json_path):
            with open(config_sentence_transformers_json_path) as fIn:
                self._model_config = json.load(fIn)
        self.__load_sbert_pooling()

    def __load_sbert_pooling(self):
        """
        It loads the pooling model from the path specified in the modules.json file.
        :return: The pooling model
        """
        model_json_path = os.path.join(self.model_path, "modules.json")
        with open(model_json_path) as fIn:
            modules_config = json.load(fIn)

        pooling_model_path = os.path.join(self.model_path, modules_config[1].get("path"))
        self.pooling_model = Pooling.load(pooling_model_path)

    def _load_auto_model(self):
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules
        """
        logger.warning(
            "No sentence-transformers model found with name {}. Creating a new one with MEAN pooling.".format(
                self.model_path
            )
        )
        model = Transformer(self.model_path)
        self.pooling_model = Pooling(model.get_word_embedding_dimension(), "mean")

    def _load_session(self):
        """
        The function loads the ONNX model into an ONNXRuntime session,
        and sets the number of threads to the number of
        logical cores on the machine
        :return: The session is being returned.
        """
        sess_options = onnxruntime.SessionOptions()

        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)

        self.session = onnxruntime.InferenceSession(
            str(self.export_model_name), sess_options, providers=[self.fast_onnxprovider]
        )
