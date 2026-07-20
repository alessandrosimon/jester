from dataclasses import dataclass
from enum import Enum, auto

class TokenKind(Enum):
    NUMBER = auto()
    IDENTIFIER = auto()

    FUNCTIONAL = auto()
    DENSITY = auto()
    KERNEL = auto()
    WEIGHTED = auto()
    LET = auto()
    PHI = auto()
    CONV = auto()
    SCALAR = auto()
    VECTOR = auto()
    
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    POWER = auto()
    
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()      
    RBRACE = auto()

    COMMA = auto()    
    COLON = auto()  
    SEMICOLON = auto()
    EQUAL = auto()

    
    EOF = auto()

@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    line: int
    column: int
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

    KEYWORDS = {
    "functional": TokenKind.FUNCTIONAL,
    "density": TokenKind.DENSITY,
    "kernel": TokenKind.KERNEL,
    "weighted": TokenKind.WEIGHTED,
    "let": TokenKind.LET,
    "phi": TokenKind.PHI,
    "conv": TokenKind.CONV,
    "scalar": TokenKind.SCALAR,
    "vector": TokenKind.VECTOR,
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
            elif char.isalpha():
                yield self.scan_identifier(start_position, start_line, start_column)
            elif self.source.startswith("**", self.position):
                self.advance()
                self.advance()
                yield Token(TokenKind.POWER, "**", start_line, start_column, start_position)
            elif char in self.SINGLE_CHAR_TOKENS:
                self.advance()
                yield Token(
                    self.SINGLE_CHAR_TOKENS[char],
                    char,
                    start_line,
                    start_column,
                    start_position
                    )
            else:
                raise SyntaxError()
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

        kind = self.KEYWORDS.get(text, TokenKind.IDENTIFIER)
        return Token(kind, text, start_line, start_column, start_position)
        #return Token(TokenKind.IDENTIFIER, text, start
        

def tokenize(src):
    t = Tokenizer(src)
    return list(t)
