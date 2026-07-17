inp = "-n0*ln(1-n3) + n1*n2/(1-n3) + 0.5*n2**3/(1-n3)**2 - n2**2*n3/(1-n3)**2 + 0.5*n2*n3**2/(1-n3)**2"


""" 
expression  := 
sum         := product (("+" | "-") product)*

product     := unary (("*" | "/") unary)*

unary       := ("+" | "-") unary
             | power

power       := primary ("**" unary)?

primary     := NUMBER
             | VARIABLE
             | FUNCTION "(" expression ")"
             | "(" expression ")"

VARIABLE    := "n0" | "n1" | "n2" | "n3"

FUNCTION    := "sin" | "cos" | "tan" | "ln" | "log" | "exp"

"""


def parse_expression(inp):
    pass

from dataclasses import dataclass
from enum import Enum, auto


class TokenKind(Enum):
    Number = auto()
    IDENTIFIER = auto()

    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    POWER = auto()
    
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()

@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    position: int

class Tokenizer:
    def __init__(self, source):
        self.source = source
        self.position = 0

    def peek(self, offset = 0):
        index  = self.position + offset
        return self.source[index]
    
    def advance(self):





