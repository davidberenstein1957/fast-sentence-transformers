import logging
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from optimum.modeling_base import OptimizedModel
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, OptimizationConfig
from transformers import AutoTokenizer, Pipeline

DEFAULT_EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-xs"

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class _SentenceEmbeddingPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs: Union[str, List[str]]) -> dict:
        """
        Preprocess the input text.

        Args:
            inputs (Union[str, List[str]]): The input text or list of texts.

        Returns:
            dict: The preprocessed inputs.
        """
        return self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

    def _forward(self, model_inputs: dict) -> dict:
        """
        Perform forward pass on the model.

        Args:
            model_inputs (dict): The model inputs.

        Returns:
            dict: The model outputs.
        """
        outputs = self.model(**model_inputs)
        return {
            "outputs": outputs,
            "attention_mask": model_inputs["attention_mask"],
        }

    def postprocess(self, model_outputs: dict) -> torch.Tensor:
        """
        Postprocess the model outputs.

        Args:
            model_outputs (dict): The model outputs.

        Returns:
            torch.Tensor: The sentence embeddings.
        """
        sentence_embeddings = self.mean_pooling(model_outputs["outputs"], model_outputs["attention_mask"])
        return F.normalize(sentence_embeddings, p=2, dim=1)

    @staticmethod
    def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform mean pooling on the model output.

        Args:
            model_output (torch.Tensor): The model output tensor.
            attention_mask (torch.Tensor): The attention mask tensor.

        Returns:
            torch.Tensor: The pooled sentence embeddings.
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class FastSentenceTransformer:
    def __init__(
        self,
        model_name_or_path: Optional[str] = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = "cpu",
        verbose: Optional[bool] = True,
    ):
        """
        FastSentenceTransformers is a class for efficient sentence embedding using transformers models.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model to use.
            device (str): The device to use for computation (e.g., "cpu", "cuda", "mps").
            verbose (bool): Whether to enable verbose logging.
        """
        self.device = device
        self.model_name_or_path = model_name_or_path
        self.onnx_cache = Path("~/.cache/onnx").expanduser()
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        if verbose:
            logger.setLevel(logging.INFO)

        if "/" not in model_name_or_path:
            if not Path(model_name_or_path).exists():
                warnings.warn(
                    "You are likely trying to use model id but you should pass the full. 'org/model-name'. In case"
                    f" you are trying to use a path, model path {model_name_or_path} not found.",
                    stacklevel=2,
                )

        self.load_model()
        self.save_model()
        self.create_pipeline()
        logger.info(f"Model loaded on {self.device}")
        logger.info("Optimizing model...")
        self.optimize_model()
        logger.info("Optimization complete.")
        self.load_optimized_model()
        logger.info("Quantizing model...")
        self.quantize_model()
        logger.info("Quantization complete...")
        self.log_model_sizes()

    def load_model(self):
        """
        Load the pre-trained model and tokenizer.
        """
        self.model: OptimizedModel = ORTModelForFeatureExtraction.from_pretrained(self.model_name_or_path, export=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def save_model(self):
        """
        Save the pre-trained model and tokenizer.
        """
        self.model.save_pretrained(self.onnx_cache)
        self.tokenizer.save_pretrained(self.onnx_cache)

    def create_pipeline(self):
        """
        Create the sentence embedding pipeline.
        """
        self.pipeline = _SentenceEmbeddingPipeline(model=self.model, tokenizer=self.tokenizer, device=self.device)

    def encode(
        self, text: Union[str, List[str]], convert_to_numpy: bool = True
    ) -> Union[List[torch.Tensor], List[np.ndarray]]:
        """
        Encode the input text into sentence embeddings.

        Args:
            text (Union[str, List[str]]): The input text or list of texts.
            convert_to_numpy (bool): Whether to convert the embeddings to NumPy arrays.

        Returns:
            Union[List[torch.Tensor], List[np.ndarray]]: The sentence embeddings.
        """
        single_input = False
        if isinstance(text, str):
            text = [text]
            single_input = True

        prediction = self.pipeline(text)
        if convert_to_numpy:
            prediction = np.array([pred.cpu().detach().numpy() for pred in prediction])

        return prediction[0][0] if single_input else prediction

    def optimize_model(self):
        """
        Optimize the model using ONNX Runtime.
        """
        optimizer = ORTOptimizer.from_pretrained(self.model)
        optimization_config = OptimizationConfig(
            optimization_level=99,
            fp16=True,
            optimize_for_gpu=False if self.device == "cpu" else True,
        )
        optimizer.optimize(
            save_dir=self.onnx_cache,
            optimization_config=optimization_config,
        )

    def load_optimized_model(self):
        """
        Load the optimized model.
        """
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_cache,
            file_name="model_optimized.onnx",
            device=self.device,
        )
        self.create_pipeline()

    def quantize_model(self):
        """
        Quantize the model using ONNX Runtime.
        """
        dynamic_quantizer = ORTQuantizer.from_pretrained(self.model)
        dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        dynamic_quantizer.quantize(
            save_dir=self.onnx_cache,
            file_suffix=f"quantized_{self.device}",
            quantization_config=dqconfig,
        )

    def load_quantized_model(self):
        """
        Load the quantized model.
        """
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_cache,
            file_name=f"quantized_optimized_{self.device}.onnx",
            device=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.onnx_cache)
        self.create_pipeline()

    def log_model_sizes(self):
        """
        Print the sizes of the model files.
        """
        size = (self.onnx_cache / "model.onnx").stat().st_size / (1024 * 1024)
        optimized_size = (self.onnx_cache / "model_optimized.onnx").stat().st_size / (1024 * 1024)
        quantized_size = (self.onnx_cache / f"model_optimized_quantized_{self.device}.onnx").stat().st_size / (
            1024 * 1024
        )
        logger.info(f"Model file size: {size:.2f} MB")
        logger.info(f"Optimized model file size: {optimized_size:.2f} MB")
        logger.info(f"Quantized and optimized model file size: {quantized_size:.2f} MB")
