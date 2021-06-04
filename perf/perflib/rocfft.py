
from typing import List, Any
from dataclasses import dataclass

@dataclass
class RIDERFFTTestRunner:
    label: str = 'UNKNOWN'
    transform: Any = None
    lengths: List[Any] = list
    ntrials: int = 1
    nbatch: int = 1
    dtype: Any = None
    verify: bool = False

    def run(self):
        print(self.label)

    def write(self, fname, results, title=None):
        pass
