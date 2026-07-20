from dataclasses import dataclass
from tokenizer import Token, TokenKind, tokenize

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
