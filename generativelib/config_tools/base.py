from collections import defaultdict
import os
from typing import Dict

from generativelib.common.utils import Utils
from generativelib.config_tools.default_values import get_default_conf
from generativelib.model.enums import ExecPhase


class ConfigReader:
    def __init__(self, config_folder_path: str, phase: ExecPhase):
        self.config = defaultdict()
        self.phase = phase
        self.conf_path = os.path.join(config_folder_path, f'{self.phase.name}_config.json')
        self.__init_config()
        
        self.global_params: Dict = self.config.get('global_params', {})
        if len(self.global_params) == 0:
            print(f'global params отсутствует в {self.conf_path}')
    
    def __create_config(self) -> None:
        print(f'Создаем файл конфигурации по умолчанию для {self.phase}')
        default_config = get_default_conf(self.phase)
        Utils.to_json(default_config, self.conf_path)
        
    def __init_config(self):
        if not os.path.exists(self.conf_path):
            self.__create_config()
        
        self.config = Utils.from_json(self.conf_path)

    def get_global_section(self, section: str) -> Dict:
        if len(self.global_params) == 0:
            return {}
            
        cur_section: Dict = self.global_params.get(section)
        if cur_section is None:
            raise ValueError('section is None')
        
        return cur_section
    
    def get_global_param_by_section(self, section: str, key: str) -> Dict:
        if len(self.global_params) == 0:
            return {}
        
        cur_section: Dict = self.global_params.get(section)
        if cur_section is None:
            raise ValueError('section is None')
        
        cur_val = cur_section.get(key, None)
        if cur_section is None:
            raise ValueError('global param is None')
        
        return cur_val