from copyreg import constructor
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
    weight: str

@dataclass(frozen=True)
class Species:
    name: str
    radius: float

@dataclass(frozen=True)
class Functional:
    name: str
    dimension: int
    species: tuple[Species, ...]
    weighted_densities: tuple[WeightedDensity, ...]
    free_energy_density: SExpr


def list_items(expr: SExpr) -> tuple[SExpr, ...]:
    if not isinstance(expr, SList):
        raise SyntaxError(f"Expected list, found {expr!r}")

    return expr.items


def symbol_name(expr: SExpr) -> str:
    if not isinstance(expr, SSymbol):
        raise SyntaxError(f"Expected symbol, found {expr!r}")

    return expr.name

def require_symbol_name(expr: SExpr, expected: str) -> None:
    actual = symbol_name(expr)

    if actual != expected:
        raise SyntaxError(
            f"Expected symbol, found {expr!r}"
        )
        
def collect_sections(expressions: tuple[SExpr, ...]
                     ) -> dict[str, tuple[SExpr, ...]]:
    sections = {}

    for expression in expressions:
        items = list_items(expression)
        head = symbol_name(items[0])

        if head in sections:
            raise ValueError(f"Duplicate section: {head}")

        sections[head] = items[1:]

    return sections
        
    
def decode_dimension(items: tuple[SExpr, ...]) -> int:
    value = items[0]
    if not isinstance(value, SNumber):
        raise SyntaxError("Dimension must be a number")

    return int(value.value)

def decode_species(items: tuple[SExpr, ...]) -> tuple[Species, ...]:
    results = []

    for expression in items:
        declaration = list_items(expression)
        name = symbol_name(declaration[0])

        radius_expr = declaration[1]

        if not isinstance(radius_expr, SNumber):
            raise SyntaxError()

        results.append(
            Species(
                name=name,
                radius=float(radius_expr.value)
                )
        )
    return tuple(results)
        

def decode_weighted_densities(
        items: tuple[SExpr, ...]
    ) -> tuple[WeightedDensity, ...]:
    results = []

    for expression in items:
        decl = list_items(expression)
        name = symbol_name(decl[0])

        constructor = list_items(decl[1])

        # for now just allow FMT constructors
        if symbol_name(constructor[0]) != "fmt-weighted-density":
            raise ValueError("Supports only FMT")

        weight = symbol_name(constructor[1])

        results.append(
            WeightedDensity(
                name=name,
                weight=weight
                )
        )
    return tuple(results)
        

def decode_free_energy(items: tuple[SExpr, ...]) -> tuple[Species, ...]:
    pass


def decode_functional(expr: SExpr) -> Functional:
    items = list_items(expr)

    require_symbol_name(items[0], "functional")
    name = symbol_name(items[1])
    sections = collect_sections(items[2:])


    return Functional(
        name=name,
        dimension=decode_dimension(sections["dimension"]),
        species=decode_species(sections["species"]),
        weighted_densities=decode_weighted_densities(
            sections["weighted-densities"]),
        free_energy_density=sections["free-energy-density"][0])
        
        




    
