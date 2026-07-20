from parser import Parser
from tokenizer import tokenize

parser = Parser("2+3*1")
parser.parse()

for t in tokenize("2+3*998"):
    print(t)
