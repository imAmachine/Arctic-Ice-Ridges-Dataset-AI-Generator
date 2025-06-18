import torch
import torchvision.transforms.v2 as T

from PIL import Image
from typing import cast

from generativelib.model.arch.enums import Modules, ModelTypes
from generativelib.model.evaluators.losses import *
from generativelib.model.inference.base import ModuleInference
from generativelib.preprocessing.processors import *

from src.infer_context import InferenceContext
from src.config_deserializer import InferenceConfigDeserializer


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
        self.generator = ModuleInference(Modules.GAN_GENERATOR, arch_module.module).to(self.device)

    def load_weights(self, path: str):
        self.generator.load_weights(path)

    def generate_from_mask(self, image: torch.Tensor) -> Image.Image:
        tensor = self._prepare_input_image(image)
        with torch.no_grad():
            generated = self.generator.generate(tensor.unsqueeze(0))
        return self._postprocess(generated, image)