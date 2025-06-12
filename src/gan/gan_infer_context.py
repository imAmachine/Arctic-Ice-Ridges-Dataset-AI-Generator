import torch
import torchvision.transforms.v2 as T

from PIL import Image
from typing import cast

from generativelib.model.arch.common_transforms import get_common_transforms
from generativelib.model.arch.enums import GenerativeModules, ModelTypes
from generativelib.model.evaluators.base import EvalItem
from generativelib.model.evaluators.enums import EvaluatorType, LossName
from generativelib.model.evaluators.losses import *
from generativelib.model.inference.base import ModuleInference
from generativelib.model.enums import ExecPhase
from generativelib.model.train.train import VisualizeHook
from generativelib.preprocessing.processors import *

from src.create_context import TrainContext, InferenceContext
from src.config_deserializer import TrainConfigDeserializer, InferenceConfigDeserializer
from src.gan.gan_templates import GanTemplate


class GanInferenceContext(InferenceContext):
    def __init__(self, config: InferenceConfigDeserializer):
        super().__init__(config)

        self._load_params()
        self._load_model()

    def _load_model(self):
        arch_module = self.config.create_arch_module(
            model_type=ModelTypes.GAN,
            module_name="gan_generator"
        )
        self.generator = ModuleInference(GenerativeModules.GAN_GENERATOR, arch_module.module).to(self.device)

    def load_weights(self, path: str):
        self.generator.load_weights(path)

    def generate_from_mask(self, image: torch.Tensor) -> Image.Image:
        tensor = self._prepare_input_image(image)
        with torch.no_grad():
            generated = self.generator.generate(tensor.unsqueeze(0))
        return self._postprocess(generated, image)