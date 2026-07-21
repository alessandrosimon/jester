from dataclasses import dataclass
from enum import Enum, auto

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
