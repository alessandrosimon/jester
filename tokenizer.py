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
