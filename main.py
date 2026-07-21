from lisp_tokenizer import tokenize
from lisp_parser import Parser
from dsl import decode_functional

#src = "(+ (* n1 n2) (/ n2 3))"

with open("rosenfeld.dft") as file:
    src = file.read()


p = Parser(src)
sexp = p.parse()

fun = decode_functional(sexp)
print(fun)

