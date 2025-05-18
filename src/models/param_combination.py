import itertools

from typing import List, Dict

class ParamGridCombination:
    def __init__(self, grid_config: Dict):
        self.grid_config = grid_config

    def generate_combination(self) -> List[Dict]:
        flat_params = self._flatten("", self.grid_config)
        keys, values = zip(*flat_params.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return [self._unflatten(combo) for combo in combinations]

    def _flatten(self, prefix: str, params: Dict) -> Dict[str, List]:
        flat = {}
        for name, value in params.items():
            full_key = f"{prefix}.{name}" if prefix else name
            if isinstance(value, dict):
                flat.update(self._flatten(full_key, value))
            else:
                flat[full_key] = value if isinstance(value, list) else [value]
        return flat

    def _unflatten(self, flat: Dict) -> Dict:
        nested = {}
        for compound_key, value in flat.items():
            keys = compound_key.split('.')
            current_level = nested
            for key in keys[:-1]:
                current_level = current_level.setdefault(key, {})
            current_level[keys[-1]] = value
        return nested