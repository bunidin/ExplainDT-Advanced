# -*- coding: utf-8 -*-
import numbers
import pyparsing as pp
from pyparsing import pyparsing_common as ppc
import functools


VERBOSE = False
if VERBOSE:
    print('Loaded modules!')


real = ppc.real
integer = ppc.integer
varname = ppc.identifier


class Variable:
    def __init__(self, varname):
        self.varname = varname

    def __repr__(self):
        return self.varname


class Classification:
    def __init__(self, cl, var):
        self.cl = cl
        self.var = var

    def __repr__(self):
        return self.cl + '(' + self.var + ')'


class Full:
    def __init__(self, var):
        self.var = var

    def __repr__(self):
        return f"(Full {str(self.var)})"


class Feature:
    def __init__(self, var, ft):
        self.var = var
        self.ft = ft

    def __repr__(self):
        return self.var + '.' + self.ft


class Not:
    def __init__(self, child):
        self.child = child

    def __repr__(self):
        return f"(Not {str(self.child)})"


class And:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"(And {str(self.left)} {str(self.right)})"


class Or:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"(Or {str(self.left)} {str(self.right)})"


class Implies:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"(Implies {str(self.left)} {str(self.right)})"


class Comparison:
    def __init__(self, sym, left, right):
        self.sym = sym
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({str(self.sym)} {str(self.left)} {str(self.right)})"


class Exists:
    def __init__(self, var, rest):
        self.var = var
        self.rest = rest

    def __repr__(self):
        return f"(Exists {str(self.var)} {str(self.rest)})"


class ForAll:
    def __init__(self, var, rest):
        self.var = var
        self.rest = rest

    def __repr__(self):
        return f"(ForAll {str(self.var)} {str(self.rest)})"


classification = pp.Group(
    varname + pp.Suppress('(') + varname + pp.Suppress(')'))
classification.setParseAction(lambda x: Classification(*x[0]))
feature = pp.Group(varname + pp.Suppress('.') + varname)
feature.setParseAction(lambda x: Feature(*x[0]))

boolean = pp.oneOf('true false')


def reducer(x, C):
    odd_list = x[0][::2]  # [a, +, b, +, c] -> [a, c]
    return functools.reduce(C, odd_list)


full = 'full' + pp.Suppress('(') + varname + pp.Suppress(')')
full.setParseAction(lambda x: Full(x[1]))

qfree = pp.infixNotation(full | classification | feature | real | integer | boolean | varname,
                         [
                             (pp.oneOf('<= > = !='), 2, pp.opAssoc.LEFT,
                              lambda x: Comparison(x[0][1], x[0][0], x[0][2])),
                             ('not', 1, pp.opAssoc.RIGHT,
                              lambda x: Not(x[0][1])),
                             ('and', 2, pp.opAssoc.LEFT,
                              lambda x: reducer(x, And)),
                             ('or', 2, pp.opAssoc.LEFT, lambda x: reducer(x, Or)),
                             ('implies', 2, pp.opAssoc.LEFT,
                              lambda x: reducer(x, Implies))
                         ])

sentence = pp.Forward()

exists = 'exists' + varname + pp.Suppress(',') + sentence
exists.setParseAction(lambda x: Exists(x[1], x[2]))

forall = 'for all' + varname + pp.Suppress(',') + sentence
forall.setParseAction(lambda x: ForAll(x[1], x[2]))

forevery = 'for every' + varname + pp.Suppress(',') + sentence
forevery.setParseAction(lambda x: ForAll(x[1], x[2]))

sentence << (exists | forall | forevery | qfree)


def arr_to_str(arr):
    # from https://stackoverflow.com/a/5445983/11780694
    return '[%s]' % ', '.join(map(str, arr))


# REPL loop
while True:
    high_query = input('(query) > ')
    if high_query in ['q', 'quit', 'exit']:
        print("bye!")
        break
    try:
        parse_results = sentence.parseString(high_query, parseAll=True)
        print(parse_results)
    except Exception as err:
        print(f" Parsing error! {err}")
