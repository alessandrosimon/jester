from dataclasses import dataclass
from enum import Enum, auto
from lisp_parser import SExpr, SList, SNumber, SSymbol

class FMTMeasure(Enum):    
    SCALAR_0 = auto()
    SCALAR_1 = auto()
    SCALAR_2 = auto()
    SCALAR_3 = auto()
    VECTOR_1 = auto()
    VECTOR_2 = auto()

@dataclass(frozen=True)
class WeightedDensity:
    name: str
    measure: FMTMeasure
    density_name: str
    radius_name: str

@dataclass(frozen=True)
class Species:
    density_name: str
    radius_name: str

@dataclass(frozen=True)
class Functional:
    name: str
    dimension: int
    species: tuple[Species, ...]
    weighted_densities: tuple[WeightedDensity, ...]
    phi: SExpr


def require_list(expr: SExpr) -> tuple[SExpr, ...]:
    if not isinstance(expr, SList):
        raise SyntaxError(f"Expected list, found {expr!r}")

    return expr.items


def require_symbol(expr: SExpr) -> str:
    if not isinstance(expr, SSymbol):
        raise SyntaxError(f"Expected symbol, found {expr!r}")

    return expr.name

def require_symbol_name(expr: SExpr, expected: str) -> None:
    actual = require_symbol(expr)

    if actual != expected:
        raise SyntaxError(
            f"Expected symbol, found {expr!r}"
        )

def decode_sections(expr):
    items = require_list(expr)

    name = require_symbol(items[0])

    if name == "dimension":
        pass
    elif name == "species":
        pass
    elif name == "weighted-densities":
        pass
    elif name == "free-energy-density":
        pass
    else:
        raise SyntaxError()
        
    
    
    
def decode_functional(expr: SExpr) -> Functional:
    items = require_list(expr)

    require_symbol_name(items[0], "functional")
    name = require_symbol(items[1])

    sections = items[2:]

    
