import argparse
from generativelib.model.arch.enums import ModelTypes
from src.config_deserializer import TrainConfigDeserializer, InferenceConfigDeserializer
from generativelib.dataset.mask_processors import *

from generativelib.model.enums import ExecPhase

from generativelib.preprocessing.processors import *
# from src.tester import ParamGridTester
from src.gui.app_start import AppStart

from src.gan.gan_context import GanTrainContext
from src.diffusion.diffusion_context import DiffusionTrainContext

configs_folder = './configs'

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--load_weights', action='store_true')
    parser.add_argument('--gui', action='store_true', help='Launch GUI interface')
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.gui:
        i_conf_deserializer = InferenceConfigDeserializer(configs_folder)
        gui_context = AppStart(i_conf_deserializer)
        gui_context.start()

    else:
        model_type = ModelTypes[args.model.upper()]
        t_conf_deserializer = TrainConfigDeserializer(configs_folder, ExecPhase.TRAIN)
        
        train_manager = None
        
        if model_type is ModelTypes.GAN:
            train_context = GanTrainContext(t_conf_deserializer, model_type)
        
        if model_type is ModelTypes.DIFFUSION:
            train_context = DiffusionTrainContext(t_conf_deserializer, model_type)
        
        train_manager = train_context.init_train(torch.device(device))
        train_manager.run(is_load_weights=args.load_weights)

if __name__ == '__main__':
    main()


# test_config = ConfigReader(configs_folder_path, ExecPhase.TEST)
# if args.test:
#     # Testing
#     print("Запуск тестов...")
#     tester = ParamGridTester(
#         param_grid_config=model_cfg,
#         trainer=trainer,
#         dataset=ds_creator,
#         output_folder_path=WEIGHTS_PATH,
#         seed=42,
#     )
#     tester.run()
    
# else:
# Training
# print(f"Обучение модели {args.model} на {args.epochs} эпохах...")
# train_template.build_train_modules(model_cfg)
# train_template._evaluators_from_config(model_cfg['evaluators_info'], device=DEVICE)

# if args.get_generator:
    #     checkpoint_map = {
    #         GenerativeModules.GENERATOR: {
    #             'model': ('trainers', GenerativeModules.GENERATOR, 'module', 'arch'),
    #         }
    #     }

    #     model = GAN(DEVICE, n_critic=5, checkpoint_map=checkpoint_map)
    #     model.build_train_modules(model_cfg)
        
    #     model.checkpoint_load(CHECKPOINT_PATH)
    #     model.checkpoint_save(os.path.join(WEIGHTS_PATH, 'generator.pt'))