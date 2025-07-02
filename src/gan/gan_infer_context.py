import torch
from PIL import Image

from generativelib.model.arch.enums import Modules, ModelTypes
from generativelib.model.evaluators.losses import *
from generativelib.model.inference.base import ModuleInference
from generativelib.preprocessing.processors import *

from src.infer_context import InferenceContext
from src.config_deserializer import InferenceConfigDeserializer


class GanInferenceContext(InferenceContext):
    def __init__(self, config: InferenceConfigDeserializer, device: torch.device):
        super().__init__(config, device)

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

    def _add_noise(self, inp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        noisy_inp = inp.clone()
        noise = torch.rand_like(inp) * 2 - 1
        noisy_inp = torch.where(mask > 0.5, noise, noisy_inp)
        return noisy_inp

    def generate_from_mask(self, image: numpy.ndarray) -> Image.Image:
        inp, mask = self._prepare_input_image(image)
        noisy_inp = self._add_noise(inp, mask)
        with torch.no_grad():
            generated = self.generator.generate(noisy_inp.unsqueeze(0))
        return self._postprocess(generated, image)