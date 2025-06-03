from abc import ABC, abstractmethod
from collections import defaultdict
import os
from typing import Any, Dict, List, Optional, Union

import torch

from generativelib.common.utils import Utils
from generativelib.config_tools.default_values import get_default_conf
from generativelib.model.arch.enums import ModelTypes
from generativelib.model.enums import ExecPhase


class ConfigReader:
    def __init__(self, config_folder: str, phase: ExecPhase):
        self.config = defaultdict()
        self.phase = phase
        self.conf_path = os.path.join(config_folder, f'{self.phase.name}_config.json')
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

    def get_global_section(self, section: str) -> Dict[str, Any]:
        if not self.global_params:
            return {}
            
        cur_section: Dict = self.global_params.get(section)
        if cur_section is None:
            return {}
        
        return cur_section
    
    def params_by_section(self, section: str, keys: Union[str, List[str]]) -> Union[Dict[str, Any] | Any]:
        if len(self.global_params) == 0:
            return {}

        cur_section: Dict = self.global_params.get(section)
        if cur_section is None:
            return {}

        if isinstance(keys, str):
            return cur_section[keys]
        
        return {key: cur_section[key] for key in keys if key in cur_section}


class ConfigModelSerializer(ABC, ConfigReader):
    def __init__(self, config_folder: str, phase: ExecPhase):
        super().__init__(config_folder, phase)
    
    @abstractmethod
    def serialize_optimize_collection(self, device: torch.device, model_type: ModelTypes):
        pass
    
    @abstractmethod
    def get_model_params(self, model_type: ModelTypes):
        pass