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
    NUMBER = auto()
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
    SINGLE_CHAR_TOKENS = {
        "+": TokenKind.PLUS,
        "-": TokenKind.MINUS,
        "/": TokenKind.SLASH,
        "*": TokenKind.STAR,
        "(": TokenKind.RPAREN,
        ")": TokenKind.LPAREN,
    }

    def __init__(self, source):
        self.source = source
        self.position = 0


    def __iter__(self):
        while not self.at_end():
            self.skip_whitespace()

            if self.at_end():
                break

            start = self.position
            char = self.peek()

            if char.isdigit():
                yield self.scan_number()
            elif char.isalpha():
                yield self.scan_identifier()
            elif self.source.startswith("**", self.position):
                self.position += 2
                yield Token(TokenKind.POWER, "**", start)
            elif char in self.SINGLE_CHAR_TOKENS:
                self.position += 1
                yield Token(
                    self.SINGLE_CHAR_TOKENS[char],
                    char,
                    start
                    )
            else:
                raise SyntaxError()
        yield Token(TokenKind.EOF, "", self.position)
            

    def peek(self, offset = 0):
        index  = self.position + offset

        if index >= len(self.source):
            return "\0"
        
        return self.source[index]
    
    def advance(self):
        char = self.peek()
        self.position += 1
        return char

    def at_end(self):
        return self.position >= len(self.source)
    

    def skip_whitespace(self):
        while self.peek().isspace():
            self.position += 1

    def scan_number(self):
        start = self.position

        while self.peek().isdigit():
            self.advance()

        if self.peek() == ".":
            self.advance()
            while self.peek().isdigit():
                self.advance()
        text = self.source[start:self.position]
        return Token(TokenKind.NUMBER, text, start)
        

    def scan_identifier(self):
        start = self.position

        while self.peek().isalnum() or self.peek() == "_":
            self.advance()

        text = self.source[start:self.position]
        return Token(TokenKind.IDENTIFIER, text, start)
        

def tokenize(src):
    t = Tokenizer(src)
    return list(t)

type Expression = Number | Variable | UnaryOp | BinaryOp | FunctionCall

@dataclass(frozen=True)
class Number:
    value: float

@dataclass(frozen=True)
class Variable:
    name: str

@dataclass(frozen=True)
class UnaryOp:
    operator: str
    operand: Expression

@dataclass(frozen=True)
class BinaryOp:
    operator: str
    left: Expression
    right: Expression


@dataclass(frozen=True)
class FunctionCall:
    name: str
    argument: Expression


class Parser:
    def __init__(self, source: str):
        self.source = source
        self.tokens = tokenize(source)
        self.index = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    def advance(self) -> Token:
        token = self.current
        self.index += 1
        return token

    def expect(self, kind: TokenKind) -> Token:
        token = self.current
        if token.kind != kind:
            raise SyntaxError(
                f"Expected {kind}, found {token.kind}"
                )
        return self.advance()

    def match(self, *kinds: TokenKind) -> Token | None:
        if self.current.kind in kinds:
            return self.advance()
        return None

    def parse(self):
        expression = self.parse_sum()
        self.expect(TokenKind.EOF)
        return expression

    def parse_sum(self) -> Expression:
        left = self.parse_product()

        while self.current.kind in {TokenKind.PLUS,
                                    TokenKind.MINUS}:
            operator = self.advance().text
            right = self.parse_product()
            left = BinaryOp(operator, left, right)

        return left

    def parse_product(self) -> Expression:
        left = self.parse_unary()

        while self.current.kind in {TokenKind.STAR,
                                    TokenKind.SLASH}:
            operator = self.advance().text
            right = self.parse_unary()
            left = BinaryOp(operator, left, right)

        return left

    def parse_unary(self) -> Expression:
        if self.current.kind in {TokenKind.PLUS, TokenKind.MINUS}:
            operator = self.advance().text
            operand = self.parse_unary()
            return UnaryOp(operator, operand)

        return self.parse_power()

    def parse_power(self) -> Expression:
        left = self.parse_primary()
        if self.match(TokenKind.POWER):
            right = self.parse_unary()
            return BinaryOp("**", left, right)

        return left

    def parse_primary(self) -> Expression:
        token = self.current

        if token.kind == TokenKind.NUMBER:
            self.advance()
            return Number(float(token.text))

        raise SyntaxError()

inp = "-n0*ln(1-n3) + n1*n2/(1-n3) + 0.5*n2**3/(1-n3)**2 - n2**2*n3/(1-n3)**2 + 0.5*n2*n3**2/(1-n3)**2"
inp = "2*3+9"


parser = Parser(inp)
parser.parse()
