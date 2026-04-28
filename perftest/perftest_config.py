import hashlib
import json
from dataclasses import dataclass, fields, asdict
from numbers import Number


@dataclass(frozen=True)
class Params:
    prec: str = "f"
    N1: Number = 320
    N2: Number = 320
    N3: Number = 1
    ntransf: int = 1
    threads: int = 1
    M: Number = 1e6
    tol: float = 1e-5

    def ndim(self) -> int:
        if self.N3 > 1:
            return 3
        if self.N2 > 1:
            return 2
        return 1

    def args(self) -> list[str]:
        return [f"{f.name}={getattr(self, f.name)}" for f in fields(self)]

    def pretty_string(self) -> str:
        n = 4
        fvalues = [f"{f.name}:{getattr(self, f.name)}" for f in fields(self)]
        chunks = [" ".join(fvalues[i : i + n]) for i in range(0, len(fvalues), n)]
        return "\n".join(chunks)

    def get_hash(self) -> str:
        # Stable short identifier used to group matching parameter sets in CSVs.
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


NRUNS = 10
PARAM_LIST = [
    Params("f", 1e4, 1, 1, 1, 1, 1e7, 1e-4),
    Params("d", 1e4, 1, 1, 1, 1, 1e7, 1e-9),
    Params("f", 320, 320, 1, 1, 1, 1e7, 1e-5),
    Params("d", 320, 320, 1, 1, 1, 1e7, 1e-9),
    Params("f", 320, 320, 1, 1, 0, 1e7, 1e-5),
    Params("d", 192, 192, 128, 1, 0, 1e7, 1e-7),
]
TRANSFORMS = [3, 2, 1]
