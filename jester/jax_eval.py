from functools import reduce
import operator

import jax
import jax.numpy as jnp

from .sexpr import SExpr, SNumber, SSymbol, SList


Array = jax.Array
Value = Array | float
Environment = dict[str, Value]

def eval_sexp(expr: SExpr, env: Environment) -> Value:
    match expr:
        case SNumber(value=val):
            return jnp.asarray(val)
        
        case SSymbol(name="pi"):
            return jnp.pi

        case SSymbol(name=name):
            try:
                return env[name]
            except:
                raise ValueError(f"Undefined symbol {name!r}") from None

        # (let ...) special form
        case SList(
                items=(
                    SSymbol(name="let"),
                    bindings_expr,
                    body_expr)
        ):
            return eval_let(bindings_expr, body_expr, env)

        case SList(
                items=(SSymbol(name=op),
                       *args_expr)):
            args = tuple(
                eval_sexp(arg, env)
                for arg in args_expr
            )

            return apply_operator(op, args)

        case SList(items=()):
            raise ValueError("Cannot evaluate empty list")
        
        case _:
            raise ValueError(f"Invalid expression: {expr!r}")

def require_list(expr: SExpr) -> tuple[SExpr, ...]:
    if not isinstance(expr, SList):
        raise ValueError(f"Expected list, got {expr!r}")

    return expr.items

def require_symbol(expr: SExpr) -> str:
    if not isinstance(expr, SSymbol):
        raise ValueError(f"Expected symbol, got {expr!r}")

    return expr.name
        
def eval_let(
        bindings_expr: SExpr,
        body_expr: SExpr,
        outer_env: Environment
) -> Value:
    bindings = require_list(bindings_expr)

    local_values: dict[str, Value] = {}

    for binding_expr in bindings:
        binding = require_list(binding_expr)

        if len(binding) != 2:
            raise ValueError(
                "let binding must have the form (name expression)")

        name = require_symbol(binding[0])
        if name in local_values:
            raise ValueError(
                "Duplicate let binding")

        # evaluate using outer environment
        local_values[name] = eval_sexp(
            binding[1],
            outer_env)

    local_env = outer_env | local_values
    return eval_sexp(body_expr, local_env)


def req_min_arity(op: str, args: tuple[Value,...], min: int) -> None:
    if len(args) < min:
        raise ValueError(f"{op}")

def req_arity(op: str, args: tuple[Value,...], expected: int) -> None:
    if len(args) != expected:
        raise ValueError(f"{op}")
    
def apply_operator(
        op: str,
        args: tuple[Value, ...],
) -> Value:
    match op:
        case "+":
            req_min_arity(op, args, 1)
            return reduce(operator.add, args)
    
        case "*":
            req_min_arity(op, args, 1)
            return reduce(operator.mul, args)

        case "-":
            req_min_arity(op, args, 1)

            if len(args) == 1:
                return -args[0]

            return reduce(operator.sub, args)

        case "/":
            req_arity(op, args, 2)
            return args[0] / args[1]

        case "pow":
            req_arity(op, args, 2)
            return jnp.power(args[0], args[1])

        case "log":
            req_arity(op, args, 1)
            return jnp.log(args[0])

        case "exp":
            req_arity(op, args, 1)
            return jnp.exp(args[0])

        case "sqrt":
            req_arity(op, args, 1)
            return jnp.sqrt(args[0])

        case "dot":
            req_arity(op, args, 2)
            return jnp.sum(
                args[0] * args[1],
                axis=0,
            )

        case _:
            raise ValueError(f"Unknown operator {op!r}")
