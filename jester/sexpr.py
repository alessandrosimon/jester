from dataclasses import dataclass
from enum import Enum, auto


####### Tokenizer code

class TokenKind(Enum):
    NUMBER = auto()
    SYMBOL = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()

@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    line: int
    column: int
    position: int

class Tokenizer:
    PARENS = {
        "(": TokenKind.LPAREN,
        ")": TokenKind.RPAREN
    }

    def __init__(self, source):
        self.source = source
        self.position = 0
        self.line = 0
        self.column = 0

    def __iter__(self):
        while not self.at_end():
            self.skip_whitespace()

            if self.at_end():
                break

            start_position = self.position
            start_line = self.line
            start_column = self.column
            
            char = self.peek()

            if char.isdigit():
                yield self.scan_number(start_position, start_line, start_column)                
            elif char in self.PARENS:
                self.advance()
                yield Token(
                    self.PARENS[char],
                    char,
                    start_line,
                    start_column,
                    start_position
                    )
            else:
                yield self.scan_symbol(start_position, start_line, start_column)
            # else:
            #     raise SyntaxError()
        yield Token(TokenKind.EOF, "", self.line, self.column, self.position)
            

    def peek(self, offset = 0):
        index  = self.position + offset

        if index >= len(self.source):
            return "\0"
        
        return self.source[index]
    
    def advance(self):
        if self.at_end():
            return "\0"
        
        char = self.source[self.position]
        self.position += 1

        if char == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1

        return char

    def at_end(self):
        return self.position >= len(self.source)
    

    def skip_whitespace(self):
        while not self.at_end() and self.peek().isspace():
            self.advance()

    def scan_symbol(self, start_position: int, start_line: int, start_column: int):
        start = self.position
        # scan until we see a space character or parens
        while (
                self.peek()
                and not self.peek().isspace() 
                and (self.peek() not in self.PARENS)
        ):
            self.advance()

        text = self.source[start:self.position]
        return Token(TokenKind.SYMBOL, text, start_line, start_column, start_position)
        

    def scan_number(self, start_position: int, start_line: int, start_column: int):
        start = self.position

        while self.peek().isdigit():
            self.advance()

        if self.peek() == ".":
            self.advance()
            while self.peek().isdigit():
                self.advance()
        text = self.source[start:self.position]
        return Token(TokenKind.NUMBER, text, start_line, start_column, start_position)
        

    def scan_identifier(self, start_position: int, start_line: int, start_column: int):
        start = self.position

        while self.peek().isalnum() or self.peek() == "_":
            self.advance()

        text = self.source[start:self.position]
        return Token(TokenKind.SYMBOL, text, start_line, start_column, start_position)

        

def tokenize(src):
    t = Tokenizer(src)
    return list(t)



######## Parser code

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
        

    def parse(self) -> SExpr:
        expression = self.parse_sexpr()
        _ = self.expect(TokenKind.EOF)

        return expression