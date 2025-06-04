import argparse
from src.gan.gan_templates import GanTemplate
from src.config_deserializer import TrainConfigDeserializer

from generativelib.model.arch.enums import ModelTypes
from generativelib.dataset.mask_processors import *
from generativelib.model.enums import ExecPhase

from generativelib.preprocessing.processors import *
# from src.tester import ParamGridTester
from src.gui.app_start import AppStart

from gan.gan_train_context import GanTrainContext

configs_folder = './configs'
t_conf_deserializer = TrainConfigDeserializer(configs_folder, ExecPhase.TRAIN)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--load_weights', action='store_true')
    parser.add_argument('--gui', action='store_true', help='Launch GUI interface')
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_type = ModelTypes[args.model.upper()]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.gui:
        gui_context = AppStart(t_conf_deserializer)
        gui_context.start()
    
    train_manager = None
    
    if model_type is ModelTypes.GAN:
        train_context = GanTrainContext(t_conf_deserializer, ModelTypes.GAN, GanTemplate)
        
        train_manager = train_context.get_train_manager(device)
        train_manager.run(is_load_weights=args.load_weights)

if __name__ == '__main__':
    main()
