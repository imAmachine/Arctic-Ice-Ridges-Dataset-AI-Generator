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


# ВРЕМЕННОЕ (видимо постоянное) РЕШЕНИЕ, НУЖНО РАЗГРЕБАТЬ!!!!!!!!!!!!!!!
class GanTrainContext(TrainContext):
    def __init__(self, config_serializer: TrainConfigDeserializer, model_type: ModelTypes):
        super().__init__(config_serializer, model_type)

    def _model_template(self) -> GanTemplate:
        model_params = self.config_serializer.model_params(ModelTypes.GAN)
        arch_collection = self.config_serializer.optimize_collection(ModelTypes.GAN)
        
        template = GanTemplate(model_params, arch_collection)
        
        return template

    def _add_model_evaluators(self, template: GanTemplate) -> None:
        """Добавляет лоссы для генератора и дискриминатора"""
        discriminator = template.get_discr_optimizer().module
        optimizers_collection = template.model_optimizers
        
        optimizers_collection.add_evals({
            GenerativeModules.GAN_GENERATOR: [
                EvalItem(
                    GeneratorLoss(discriminator),
                    name=LossName.ADVERSARIAL.name,
                    type=EvaluatorType.LOSS,
                    weight=1.0
                )
            ],
            GenerativeModules.GAN_DISCRIMINATOR: [
                EvalItem(
                    WassersteinLoss(discriminator),
                    name=LossName.WASSERSTEIN.name,
                    type=EvaluatorType.LOSS,
                    weight=1.0
                ),
                EvalItem(
                    GradientPenalty(discriminator),
                    name=LossName.GRADIENT_PENALTY.name,
                    type=EvaluatorType.LOSS,
                    weight=10.0,
                    exec_phase=ExecPhase.TRAIN
                )
            ]
        })
    
    def _get_model_transform(self, img_size: int) -> T.Compose:
        transforms = get_common_transforms(cast(int, img_size))
        return transforms
    
    def _visualize_for_model(self, template: GanTemplate) -> VisualizeHook:
        return self._visualize_hook(
            gen_callable=template.get_gen_optimizer().module
        )