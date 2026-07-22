from lisp_tokenizer import tokenize
from lisp_parser import Parser
from dsl import decode_functional
from eval_jax import eval_sexp

#src = "(+ (* n1 n2) (/ n2 3))"



with open("test.dft") as file:
    src = file.read()


p = Parser(src)
sexp = p.parse()

fun = decode_functional(sexp)

print(eval_sexp(fun.free_energy_density, {}))

