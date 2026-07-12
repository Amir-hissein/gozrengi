from dataclasses import dataclass

import numpy as np


@dataclass
class RenkSonucu:
    ad: str
    hex_kodu: str
    skor: float
    rgb: tuple
    hsv: tuple


@dataclass
class GozBilgisi:
    taraf: str
    iris_goruntu: np.ndarray
    ortalama_rgb: tuple
    renk_sonucu: RenkSonucu
    iris_merkez: tuple
