from dataclasses import dataclass
from lisp_tokenizer import tokenize, Token, TokenKind

class SExpr:
    pass

@dataclass(frozen=True)
class SNumber(SExpr):
    value: float

@dataclass(frozen=True)
class SSymbol(SExpr):
    name: str

@dataclass(frozen=True)
class SList(SExpr):
    items: tuple[SExpr, ...]
    
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

    def parse_sexpr(self) -> SExpr:
        token = self.current

        if token.kind == TokenKind.EOF:
            raise SyntaxError("Unexpected EOF")

        if token.kind == TokenKind.NUMBER:
            self.advance()
            return SNumber(float(token.text))

        if token.kind == TokenKind.SYMBOL:
            self.advance()
            return SSymbol(token.text)

        if token.kind == TokenKind.LPAREN:
            self.advance()

            items = []

            while self.current.kind != TokenKind.RPAREN:
                if self.current.kind == TokenKind.EOF:
                    raise SyntaxError("Unclosed list")

                items.append(self.parse_sexpr())

            self.expect(TokenKind.RPAREN)
            return SList(tuple(items))

        
        if token.kind == TokenKind.RPAREN:
            raise SyntaxError("Unexpected ')'")
        
        raise SyntaxError(f"Unexpected token: {token.text!r}")
        

    def parse(self) -> tuple[SExpr, ...]:
        expressions = []
        while self.current.kind != TokenKind.EOF:
            expressions.append(self.parse_sexpr())

        return tuple(expressions)
