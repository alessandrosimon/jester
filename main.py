from lisp_tokenizer import tokenize
from lisp_parser import Parser

src = "(+ (* n1 n2) (/ n2 3))"

for t in tokenize(src):
    print(t)

p = Parser(src)
code = p.parse()

for c in code:
    print(c)
