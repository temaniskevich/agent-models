"""abm_banks — агентная модель банковского сектора России как генератор
синтетических данных (месячный шаг, векторизованная, воспроизводимая)."""
from .config import ModelConfig, preset, PRESET_LABELS
from .model import BankingModel, SimulationResult
from .scenarios import Scenario, build_scenario

__all__ = ["ModelConfig", "BankingModel", "SimulationResult", "Scenario",
           "build_scenario", "preset", "PRESET_LABELS"]
__version__ = "2.0.0"
