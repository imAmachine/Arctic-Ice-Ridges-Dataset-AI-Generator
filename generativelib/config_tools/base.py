from collections import defaultdict
import os
from typing import Any, Dict, List, Union

from generativelib.common.utils import Utils
from generativelib.config_tools.default_values import get_default_conf
from generativelib.model.enums import ExecPhase


class ConfigReader:
    def __init__(self, config_folder: str, phase: ExecPhase):
        self.phase = phase
        
        os.makedirs(config_folder, exist_ok=True)
        
        self.conf_path = os.path.join(config_folder, f'{self.phase.name}_config.json')
        
        self.config: Dict[str, Any] = {}
        self.__init_config()
        
        self.global_params: Dict[str, Any] = self.config.get('global_params', {})
        
        if not self.global_params:
            print(f'global params отсутствует в {self.conf_path}')
    
    def __create_config(self) -> None:
        print(f'Создаем файл конфигурации по умолчанию для {self.phase}')
        default_config = get_default_conf(self.phase)
        Utils.to_json(default_config, self.conf_path)
        
    def __init_config(self) -> None:
        if not os.path.exists(self.conf_path):
            self.__create_config()

        self.config = Utils.from_json(self.conf_path)

    def get_global_section(self, section: str) -> Dict[str, Any]:
        return self.global_params.get(section, {})
    
    def params_by_section(self, section: str, keys: Union[str, List[str]]) -> Union[Any, Dict[str, Any]]:
        cur_section: Dict[str, Any] = self.global_params.get(section, {})

        if isinstance(keys, str):
            return cur_section.get(keys)

        return {
            key: cur_section.get(key)
            for key in keys 
            if key in cur_section
        }
