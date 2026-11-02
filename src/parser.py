"""
NEUROGEN Parser v0.2 — supports expressions, LET, CONST, typed INPUT, operators.
"""

import re
from typing import List, Tuple, Any, Union
from ast_nodes import *


class ParseError(Exception):
    def __init__(self, message: str, line: int = 0):
        super().__init__(f"Line {line}: {message}")
        self.line = line


# Keywords (uppercase by convention)
KEYWORDS = {
    "GENOME", "MEMORY", "STRUCTURE", "MOTIF", "INPUT", "CONST", "LET",
    "TRANSFORM", "CONNECT", "EVALUATE", "OBSERVE", "ADAPT", "OUTPUT", "USE",
    "ASSOCIATIVE", "KEYVALUE",
    # v0.3 pipeline keywords
    "PIPELINE", "TO",
    # types
    "BOOL", "INT", "FLOAT", "STRING", "VECTOR", "MATRIX", "TENSOR", "OBJECT",
}


def tokenize(source: str) -> List[Tuple[str, Any, int]]:
    tokens = []
    lines = source.splitlines()
    for lineno, line in enumerate(lines, 1):
        if "#" in line:
            line = line[: line.index("#")]
        line = line.strip()
        if not line:
            continue

        pattern = r'''
            "([^"\\]*(\\.[^"\\]*)*)"   |
            '([^'\\]*(\\.[^'\\]*)*)'   |
            -?\d+\.\d+                 |
            -?\d+                      |
            [A-Za-z_][A-Za-z0-9_]*     |
            [(){}\[\],=:.]             |
            \S
        '''
        for m in re.finditer(pattern, line, re.VERBOSE):
            p = m.group(0)
            if p.startswith('"') or p.startswith("'"):
                tokens.append(("STRING", p[1:-1], lineno))
            elif re.fullmatch(r"-?\d+\.\d+", p):
                tokens.append(("NUMBER", float(p), lineno))
            elif re.fullmatch(r"-?\d+", p):
                tokens.append(("NUMBER", int(p), lineno))
            elif p in "(){}[],=:.":
                tokens.append(("PUNCT", p, lineno))
            elif p in KEYWORDS or p.upper() in {"ASSOCIATIVE", "KEYVALUE"}:
                tokens.append(("KEYWORD", p.upper(), lineno))
            else:
                tokens.append(("IDENT", p, lineno))
    return tokens


class Parser:
    def __init__(self, tokens: List[Tuple[str, Any, int]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def expect(self, typ: str, value: Any = None):
        tok = self.advance()
        if tok is None:
            raise ParseError(f"Expected {typ} but reached end of input")
        if tok[0] != typ or (value is not None and tok[1] != value):
            raise ParseError(f"Expected {typ} {value or ''}, got {tok}", tok[2])
        return tok

    def match(self, typ: str, value: Any = None) -> bool:
        tok = self.peek()
        if tok and tok[0] == typ and (value is None or tok[1] == value):
            self.advance()
            return True
        return False

    def parse(self) -> Genome:
        genome_name = "Anonymous"
        statements = []

        while self.peek():
            tok = self.peek()
            if tok[1] == "GENOME":
                self.advance()
                name_tok = self.expect("STRING")
                genome_name = name_tok[1]
            elif tok[1] == "MEMORY":
                statements.append(self.parse_memory())
            elif tok[1] == "STRUCTURE":
                statements.append(self.parse_structure())
            elif tok[1] == "MOTIF":
                statements.append(self.parse_motif())
            elif tok[1] == "OUTPUT":
                statements.append(self.parse_output())
            elif tok[1] == "PIPELINE":
                statements.append(self.parse_pipeline())
            else:
                raise ParseError(f"Unexpected token {tok[1]}", tok[2])

        return Genome(name=genome_name, statements=statements)

    def parse_memory(self) -> MemoryDecl:
        self.expect("KEYWORD", "MEMORY")
        kind_tok = self.advance()
        if kind_tok[1] not in ("ASSOCIATIVE", "KEYVALUE"):
            raise ParseError("MEMORY must be 'associative' or 'keyvalue'", kind_tok[2])
        return MemoryDecl(kind=kind_tok[1].lower())

    def parse_structure(self) -> Structure:
        self.expect("KEYWORD", "STRUCTURE")
        name = self.expect("IDENT")[1]
        self.expect("PUNCT", "{")
        body = self.parse_body()
        self.expect("PUNCT", "}")
        return Structure(name=name, body=body)

    def parse_motif(self) -> Motif:
        self.expect("KEYWORD", "MOTIF")
        name = self.expect("IDENT")[1]
        self.expect("PUNCT", "{")
        body = self.parse_body()
        self.expect("PUNCT", "}")
        return Motif(name=name, body=body)

    def parse_body(self) -> List[Node]:
        body = []
        while self.peek() and not (self.peek()[0] == "PUNCT" and self.peek()[1] == "}"):
            tok = self.peek()
            kw = tok[1] if tok[0] == "KEYWORD" else None

            if kw == "INPUT":
                body.append(self.parse_input())
            elif kw == "CONST":
                body.append(self.parse_const())
            elif kw == "LET":
                body.append(self.parse_let())
            elif kw == "TRANSFORM":
                self.advance()
                ident = self.expect("IDENT")[1]
                body.append(TransformStmt(identifier=ident))
            elif kw == "CONNECT":
                self.advance()
                ident = self.expect("IDENT")[1]
                body.append(ConnectStmt(identifier=ident))
            elif kw == "EVALUATE":
                self.advance()
                ident = self.expect("IDENT")[1]
                body.append(EvaluateStmt(identifier=ident))
            elif kw == "OBSERVE":
                self.advance()
                ident = self.expect("IDENT")[1]
                body.append(ObserveStmt(identifier=ident))
            elif kw == "ADAPT":
                self.advance()
                target = self.expect("IDENT")[1]
                body.append(AdaptStmt(target=target))
            elif kw == "USE":
                self.advance()
                motif = self.expect("IDENT")[1]
                body.append(UseStmt(motif_name=motif))
            elif kw == "OUTPUT":
                # v0.3: named output declaration inside a STRUCTURE body
                body.append(self.parse_structure_output())
            elif tok[0] == "IDENT" or (tok[0] == "KEYWORD" and tok[1] in ("VECTOR", "MATRIX")):
                body.append(self.parse_possible_call())
            else:
                raise ParseError(f"Unexpected statement inside structure/motif: {tok}", tok[2])
        return body

    def parse_input(self) -> InputStmt:
        self.expect("KEYWORD", "INPUT")
        ident = self.expect("IDENT")[1]
        type_name = None
        if self.match("PUNCT", ":"):
            type_tok = self.advance()
            type_name = type_tok[1]
        return InputStmt(identifier=ident, type_name=type_name)

    def parse_const(self) -> ConstStmt:
        self.expect("KEYWORD", "CONST")
        ident = self.expect("IDENT")[1]
        self.expect("PUNCT", "=")
        val_tok = self.advance()
        if val_tok[0] == "NUMBER":
            value = val_tok[1]
        elif val_tok[0] == "STRING":
            value = val_tok[1]
        else:
            raise ParseError(f"Expected literal for CONST, got {val_tok}", val_tok[2])
        return ConstStmt(identifier=ident, value=value)

    def parse_let(self) -> LetStmt:
        self.expect("KEYWORD", "LET")
        ident = self.expect("IDENT")[1]
        self.expect("PUNCT", "=")
        tok = self.peek()
        # v0.3: KEYWORD accepted here only for registered operators whose
        # names collide with type keywords (VECTOR, MATRIX); anything else
        # stays a parse error as in v0.2.1.
        if tok[0] == "IDENT" or (tok[0] == "KEYWORD" and tok[1] in ("VECTOR", "MATRIX")):
            name = self.advance()[1]
            if self.peek() and self.peek()[0] == "PUNCT" and self.peek()[1] == "(":
                self.advance()  # (
                args = self.parse_arg_list()
                self.expect("PUNCT", ")")
                return LetStmt(identifier=ident, operator=name, args=args)
            else:
                return LetStmt(identifier=ident, operator=None, args=[name])
        elif tok[0] == "NUMBER" or tok[0] == "STRING":
            val = self.advance()[1]
            return LetStmt(identifier=ident, operator=None, args=[val])
        else:
            raise ParseError(f"Invalid LET right-hand side: {tok}", tok[2])

    def parse_possible_call(self) -> Node:
        # v0.3: VECTOR/MATRIX accepted as bare calls (type keywords that are
        # also registered operators).
        tok = self.peek()
        if tok[0] == "KEYWORD" and tok[1] in ("VECTOR", "MATRIX"):
            name = self.advance()[1]
        else:
            name = self.expect("IDENT")[1]
        if self.peek() and self.peek()[0] == "PUNCT" and self.peek()[1] == "(":
            self.advance()
            args = self.parse_arg_list()
            self.expect("PUNCT", ")")
            return CallStmt(operator=name, args=args)
        return CallStmt(operator="IDENTITY", args=[name])

    def parse_arg_list(self) -> List[Any]:
        args = []
        if self.peek() and self.peek()[0] == "PUNCT" and self.peek()[1] == ")":
            return args
        while True:
            tok = self.peek()
            if tok[0] in ("NUMBER", "STRING"):
                args.append(self.advance()[1])
            elif tok[0] == "IDENT" or tok[0] == "KEYWORD":
                args.append(self.advance()[1])
            else:
                raise ParseError(f"Expected argument, got {tok}", tok[2] if tok else 0)
            if self.match("PUNCT", ","):
                continue
            break
        return args

    def parse_output(self) -> OutputStmt:
        self.expect("KEYWORD", "OUTPUT")
        ident = self.expect("IDENT")[1]
        return OutputStmt(identifier=ident)

    # ---- v0.3 pipeline syntax ----

    def parse_structure_output(self) -> StructureOutput:
        """OUTPUT <name> inside a STRUCTURE body."""
        self.expect("KEYWORD", "OUTPUT")
        name = self.expect("IDENT")[1]
        return StructureOutput(name=name)

    def parse_dotted_ref(self) -> Tuple[str, str]:
        """Parse <ident>.<ident> (e.g. perception.features)."""
        first = self.expect("IDENT")[1]
        self.expect("PUNCT", ".")
        second = self.expect("IDENT")[1]
        return first, second

    def parse_pipeline(self) -> Pipeline:
        self.expect("KEYWORD", "PIPELINE")
        name = self.expect("IDENT")[1]
        self.expect("PUNCT", "{")
        connects: List[PipelineConnect] = []
        outputs: List[Tuple[str, str]] = []
        while self.peek() and not (self.peek()[0] == "PUNCT" and self.peek()[1] == "}"):
            tok = self.peek()
            kw = tok[1] if tok[0] == "KEYWORD" else None
            if kw == "CONNECT":
                connects.append(self.parse_pipeline_connect())
            elif kw == "OUTPUT":
                outputs.append(self.parse_pipeline_output())
            else:
                raise ParseError(
                    f"Unexpected statement inside PIPELINE: {tok[1]} (expected CONNECT or OUTPUT)",
                    tok[2],
                )
        self.expect("PUNCT", "}")
        if not outputs:
            raise ParseError(f"PIPELINE '{name}' must declare at least one OUTPUT", 0)
        return Pipeline(name=name, connects=connects, outputs=outputs)

    def parse_pipeline_connect(self) -> PipelineConnect:
        self.expect("KEYWORD", "CONNECT")
        src_struct, src_output = self.parse_dotted_ref()
        self.expect("KEYWORD", "TO")
        dst_struct, dst_input = self.parse_dotted_ref()
        return PipelineConnect(
            src_struct=src_struct,
            src_output=src_output,
            dst_struct=dst_struct,
            dst_input=dst_input,
        )

    def parse_pipeline_output(self) -> Tuple[str, str]:
        self.expect("KEYWORD", "OUTPUT")
        return self.parse_dotted_ref()


def parse(source: str) -> Genome:
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse()
