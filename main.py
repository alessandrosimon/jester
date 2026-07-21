from lisp_tokenizer import tokenize
from lisp_parser import Parser

#src = "(+ (* n1 n2) (/ n2 3))"

with open("rosenfeld.dft") as file:
    src = file.read()

for t in tokenize(src):
    print(t)

p = Parser(src)
code = p.parse()

for c in code:
    print(c)
