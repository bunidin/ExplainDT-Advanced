# -*- coding: utf-8 -*-
import pyparsing as pp
from pyparsing import ParseBaseException, pyparsing_common as ppc
import functools
from colorist import Color, ColorHex
from sklearn.externals._packaging.version import parse
from components.predicates.completion_predicates import AllPoss
import context.model as Model
from components import eval_formula
import components.predicates as Pred
import components.operators as Operators
import components.instances as Instances
import sys
from context import Symbol
import os
import random

if len(sys.argv) < 2:
    print('Usage: python3 interpreter.py <path_to_sat_solver>')
    exit(1)
sat_solver_path = sys.argv[1]

STATE = {'model_filename': None, 'model_object': None}

def load_file(filename):
    STATE['model_filename'] = filename
    STATE['model_object'] = Model.Tree(from_file=filename)
    print(f"{Color.YELLOW}Tree successfully loaded from{Color.OFF} '{filename}'")

if len(sys.argv) > 2:
    model_filename = sys.argv[2]
    load_file(model_filename)

VERBOSE = False
if VERBOSE:
    print('Loaded modules!')

reserved_words = pp.oneOf('true false full all_pos all_neg LEL RFS node SR cons load show relevant generate')

varname = ~reserved_words + ppc.identifier
varname.setParseAction(lambda x: Variable(x[0]))

f_values = pp.oneOf('true false 0 1 ?')
constant = pp.delimitedList(f_values, delim=',')
constant = constant | pp.Suppress('[') + pp.delimitedList(f_values, delim=',') + pp.Suppress(']')

constant.setParseAction(lambda x: Constant(x))

class Variable:
    def __init__(self, varname):
        self.varname = varname

    def __repr__(self):
        return f"(Var {str(self.varname)})"

    def to_api(self):
        return Instances.Var(self.varname)

class Constant:
    def __init__(self, constant) -> None:
        self.constant = constant

    def __repr__(self) -> str:
        return f"(Constant {str(self.constant)})"

    def to_api(self):
        def str_to_sym(s):
            if s == '0' or s == 'false':
                return Symbol.ZERO
            elif s == '1' or s == 'true':
                return Symbol.ONE
            else:
                return Symbol.BOT
        sym_tuple = tuple(list(map(str_to_sym, self.constant)))
        return Instances.Constant(sym_tuple)

class Classification:
    def __init__(self, cl, vr):
        self.cl = cl
        self.vr = vr

    def __repr__(self):
        return f"(Classification {str(self.cl)} {str(self.vr)})"

    def to_api(self):
        if self.cl == STATE['model_object'].classes[0]:
            return Operators.And(Pred.Full(self.vr.to_api()), Pred.AllPoss(self.vr.to_api()))
        elif self.cl == STATE['model_object'].classes[1]:
            return Operators.And(Pred.Full(self.vr.to_api()), Pred.AllNeg(self.vr.to_api()))
        else:
            raise Exception(f"Class {self.cl} is not in the model classes ({STATE['model_object'].classes})")

class Feature:
    def __init__(self, vr, ft_name):
        self.vr = vr
        self.ft_name = ft_name

    def __repr__(self):
        return f"(Feature {str(self.vr)} {str(self.ft_name)})"

class Relevant:
    def __init__(self, ft) -> None:
        self.ft = ft

    def __repr__(self) -> str:
        return f"(Relevant {str(self.ft)})"

    def to_api(self):
        constant_array = ['?' if feature == self.ft else '1' for feature in STATE['model_object'].features]
        ct = Constant(constant_array)
        return Operators.Not(Pred.RFS(ct.to_api()))

relevant = pp.Suppress('relevant') + pp.Word(pp.alphanums)
relevant.setParseAction(lambda x: Relevant(x[0]))

class Full:
    def __init__(self, vr):
        self.vr = vr

    def __repr__(self):
        return f"(Full {str(self.vr)})"

    def to_api(self):
        return Pred.Full(self.vr.to_api())

class Not:
    def __init__(self, child):
        self.child = child

    def __repr__(self):
        return f"(Not {str(self.child)})"

    def to_api(self):
        return Operators.Not(self.child.to_api())

class And:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"(And {str(self.left)} {str(self.right)})"

    def to_api(self):
        return Operators.And(self.left.to_api(), self.right.to_api())

class Or:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"(Or {str(self.left)} {str(self.right)})"

    def to_api(self):
        return Operators.Or(self.left.to_api(), self.right.to_api())

class Implies:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"(Implies {str(self.left)} {str(self.right)})"

    def to_api(self):
        return Operators.Or(Operators.Not(self.left.to_api()), self.right.to_api())

class Subsumption:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"(Subsumption {str(self.left)} {str(self.right)})"

    def to_api(self):
        return Pred.Subsumption(self.left.to_api(), self.right.to_api())

class Exists:
    def __init__(self, var, rest):
        self.var = var
        self.rest = rest

    def __repr__(self):
        return f"(Exists {str(self.var)} {str(self.rest)})"

    def to_api(self):
        return Operators.Exists(self.var.to_api(), self.rest.to_api())

class ForAll:
    def __init__(self, var, rest):
        self.var = var
        self.rest = rest

    def __repr__(self):
        return f"(ForAll {str(self.var)} {str(self.rest)})"

    def to_api(self):
        return Operators.Not(Operators.Exists(self.var.to_api(), self.rest.to_api()))

class AllPos:
    def __init__(self, var_or_const):
        self.var_or_const = var_or_const

    def __repr__(self) -> str:
        return f"(AllPos {str(self.var_or_const)})"

    def to_api(self):
        return Pred.AllPoss(self.var_or_const.to_api())

class AllNeg:
    def __init__(self, var_or_const):
        self.var_or_const = var_or_const

    def __repr__(self) -> str:
        return f"(AllNeg {str(self.var_or_const)})"

    def to_api(self):
        return Pred.AllNeg(self.var_or_const.to_api())

class Node:
    def __init__(self, var_or_const):
        self.var_or_const = var_or_const

    def __repr__(self) -> str:
        return f"(Node {str(self.var_or_const)})"

    def to_api(self):
        return Pred.IsNode(self.var_or_const.to_api())

class RFS:
    def __init__(self, var_or_const):
        self.var_or_const = var_or_const

    def __repr__(self) -> str:
        return f"(RFS {str(self.var_or_const)})"

    def to_api(self):
        return Pred.RFS(self.var_or_const.to_api())

class SR:
    def __init__(self, var_or_const1, var_or_const2):
        self.var_or_const1 = var_or_const1
        self.var_or_const2 = var_or_const2

    def __repr__(self) -> str:
        return f"(SR {str(self.var_or_const1)} {str(self.var_or_const2)})"

    def to_api(self):
        return And(Full(self.var_or_const1),
                   And(Subsumption(self.var_or_const2, self.var_or_const1),
                       Or(And(AllPos(self.var_or_const1), AllPos(self.var_or_const2)),
                          And(AllNeg(self.var_or_const1), AllNeg(self.var_or_const2))))).to_api()

class LEL:
    def __init__(self, var_or_const1, var_or_const2):
        self.var_or_const1 = var_or_const1
        self.var_or_const2 = var_or_const2

    def __repr__(self) -> str:
        return f"(LEL {str(self.var_or_const1)} {str(self.var_or_const2)})"

    def to_api(self):
        return Pred.LEL(self.var_or_const1.to_api(), self.var_or_const2.to_api())

class Cons:
    def __init__(self, var_or_const1, var_or_const2):
        self.var_or_const1 = var_or_const1
        self.var_or_const2 = var_or_const2

    def __repr__(self) -> str:
        return f"(Cons {str(self.var_or_const1)} {str(self.var_or_const2)})"

    def to_api(self):
        return Pred.Cons(self.var_or_const1.to_api(), self.var_or_const2.to_api())

class Action:
    def __init__(self, action, *args):
        self.action = action
        self.args = args

    def __repr__(self):
        return f"(Action {str(self.action)} {str(self.args)})"

    def execute(self):
        return self.action(*self.args)

var_or_const = varname | constant

def reducer(x, C):
    odd_list = x[0][::2]  # [a, +, b, +, c] -> [a, c]
    return functools.reduce(C, odd_list)

def unary_op(op, op_class):
    result = op + pp.Suppress('(') + var_or_const + pp.Suppress(')')
    result.setParseAction(lambda x: op_class(x[1]))
    return result

def binary_op(op, op_class):
    result = op + pp.Suppress('(') + var_or_const + \
        pp.Suppress(',') + var_or_const + pp.Suppress(')')
    result.setParseAction(lambda x: op_class(x[1], x[2]))
    return result

class NoModelError(Exception):
    def __init__(self, message="This action/query requires a loaded model!"):
        self.message = message
        super().__init__(self.message)

def fun_show_classes():
    if STATE['model_object'] is None:
        raise NoModelError
    classes = STATE['model_object'].classes
    answer = f"{Color.YELLOW}Classes{Color.OFF}:\n"
    for i, cl in enumerate(classes):
        answer += f"{i+1}) {cl}\n"
    print(answer[:-1])

def fun_show_features():
    if STATE['model_object'] is None:
        raise NoModelError
    features = STATE['model_object'].features
    answer = f"{Color.YELLOW}Features{Color.OFF} ({len(features)} total):\n"
    for i, ft in enumerate(features):
        answer += f"{i+1}) {ft}\n"
    print(answer[:-1])

# Instance generation functions
def generate_random_instance(length):
    sequence = [random.choice([0, 1]) for _ in range(length)]
    return sequence

def generate_random_partial_instance(length):
    sequence = [random.choice([0, 1, '?']) for _ in range(length)]
    return sequence

def fun_generate(length):
    try:
        length = int(length)
        full_instance = generate_random_instance(length)
        partial_instance = generate_random_partial_instance(length)
        formatted_full = ','.join(map(str, full_instance))
        formatted_partial = ','.join(map(str, partial_instance))
        print(f'Full instance: ({formatted_full})')
        print(f'Partial instance: ({formatted_partial})')
    except ValueError:
        print(f"{Color.RED}Invalid length value!{Color.OFF}")

# keywords for showing
show_classes = pp.Keyword('show classes')
show_features = pp.Keyword('show features')
# loading
load = 'load' + pp.Word(pp.printables)
generate = 'generate' + ppc.integer  # Add generate keyword

show_classes.setParseAction(lambda _: Action(fun_show_classes))
show_features.setParseAction(lambda _: Action(fun_show_features))
load.setParseAction(lambda x: Action(load_file, x[1]))
generate.setParseAction(lambda x: Action(fun_generate, x[1]))  # Set parse action for generate

action = show_classes | show_features | load | generate  # Include generate in action

# UNARY predicates
full = unary_op('full', Full)
all_pos = unary_op('all_pos', AllPos)
all_neg = unary_op('all_neg', AllNeg)
rfs = unary_op('RFS', RFS)
node = unary_op('node', Node)

unary_pred = full | all_pos | all_neg | rfs | node

# BINARY predicates
subsumption = pp.infixNotation(var_or_const,
                               [
                                        ('subsumed by', 2, pp.opAssoc.LEFT,
                                            lambda x: reducer(x, Subsumption)),
                                        ('is subsumed by', 2, pp.opAssoc.LEFT,
                                            lambda x: reducer(x, Subsumption)),
                                        ('<=', 2, pp.opAssoc.LEFT,
                                            lambda x: reducer(x, Subsumption)),
                               ])

cons = binary_op('cons', Cons)
lel = binary_op('LEL', LEL)
sr = binary_op('SR', SR)

binary_pred = subsumption | cons | lel | sr

# Classification
class_name = pp.Word(pp.alphanums + ' ')
class_name_with_quotes = pp.quotedString.setParseAction(pp.removeQuotes)
class_name_parser = class_name_with_quotes | class_name

classification = class_name_parser + pp.Suppress('(') + var_or_const + pp.Suppress(')')
classification.setParseAction(lambda x: Classification(x[0], x[1]))

# classification.parse_string('(x)')

feature = varname + pp.Suppress('.') + pp.Word(pp.alphanums)
feature.setParseAction(lambda x: Feature(x[0], x[1]))

feature_test = pp.infix_notation(feature | f_values,
                                 [
                                     ('=', 2, pp.opAssoc.LEFT,
                                      lambda x: reducer(x, feature_test_to_subsumption))
                                 ])

def feature_test_to_subsumption(feature_obj, f_value):
    if STATE['model_object'] is None:
        raise NoModelError
    features = STATE['model_object'].features
    ft_name = feature_obj.ft_name
    vr = feature_obj.vr
    const = Constant([f_value if feature == ft_name else '?' for feature in features])
    return Subsumption(const, vr)

# feature_test.setParseAction(lambda x: feature_test_to_subsumption(x[0], x[1], x[2]))

atomic = unary_pred | classification | feature_test | relevant | binary_pred 

qfree = pp.infixNotation(atomic,
                         [
                             ('not', 1, pp.opAssoc.RIGHT,
                              lambda x: Not(x[0][1])),
                             ('and', 2, pp.opAssoc.LEFT,
                              lambda x: reducer(x, And)),
                             ('or', 2, pp.opAssoc.LEFT,
                              lambda x: reducer(x, Or)),
                             ('implies', 2, pp.opAssoc.LEFT,
                              lambda x: reducer(x, Implies))
                         ])

sentence = pp.Forward()
query = pp.Forward()

exists = 'exists' + varname + pp.Suppress(',') + query
exists.setParseAction(lambda x: Exists(x[1], x[2]))

forall = 'for all' + varname + pp.Suppress(',') + query
forall.setParseAction(lambda x: ForAll(x[1], x[2]))

sentence = sentence << (exists | forall | qfree)

query = action | query << pp.infix_notation(sentence,
                                   [
                                       ('not', 1, pp.opAssoc.RIGHT,
                                        lambda x: Not(x[0][1])),
                                       ('and', 2, pp.opAssoc.LEFT,
                                        lambda x: reducer(x, And)),
                                       ('or', 2, pp.opAssoc.LEFT,
                                        lambda x: reducer(x, Or)),
                                       ('implies', 2, pp.opAssoc.LEFT,
                                        lambda x: reducer(x, Implies))
                                   ])

def arr_to_str(arr):
    # from https://stackoverflow.com/a/5445983/117806 94
    return '[%s]' % ', '.join(map(str, arr))

# Main execution function 
def solve(query):
    if STATE['model_object'] is None:
        raise NoModelError
    dimension = len(STATE['model_object'].features)
    flip_value = isinstance(query, ForAll)
    api_query = query.to_api()
    truth_value = eval_formula(api_query, dimension, STATE['model_filename'], sat_solver_path)
    if flip_value: 
        truth_value = not truth_value
    ORANGE = ColorHex("#ff5733")
    if truth_value:
        return f"{Color.GREEN}Yes{Color.OFF}"
    else:
        return f"{ORANGE}NO{Color.OFF}"

# At the start of the REPL loop, check if running interactively
interactive_mode = os.isatty(0)  # Check if stdin is a TTY (interactive terminal)

# REPL loop
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"{Color.CYAN} ExplainDT {Color.OFF} v1.01 modified by Buni")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
while interactive_mode:
    try:
        high_query = input(f'\n{Color.GREEN}>> {Color.OFF}')
        if high_query in ['q', 'quit', 'exit']:
            print("bye!")
            break
        if len(high_query) == 0:
            continue
        if high_query[0] == '#':
            print(f'{Color.CYAN}comment{Color.OFF}: {ColorHex("#bbbbbb")}{high_query[1:]}{Color.OFF}')
            continue
        try:
            parse_results = query.parseString(high_query, parseAll=True)[0]
            if isinstance(parse_results, Action):
                parse_results.execute()
            else:
                print(f'{Color.CYAN}out{Color.OFF}: {solve(parse_results)}')
        except ParseBaseException as err:
            print(f"{Color.RED}Parsing error!{Color.OFF}: {err}")
        except Exception as err:
            print(f"{Color.RED}Execution error!{Color.OFF}: {err}")
    except EOFError:
        break

# Add code to handle the case when running non-interactively
if not interactive_mode:
    try:
        high_query = sys.stdin.read().strip()
        if len(high_query) == 0:
            sys.exit(0)
        if high_query[0] == '#':
            print(f'{Color.CYAN}comment{Color.OFF}: {ColorHex("#bbbbbb")}{high_query[1:]}{Color.OFF}')
            sys.exit(0)
        try:
            parse_results = query.parseString(high_query, parseAll=True)[0]
            if isinstance(parse_results, Action):
                parse_results.execute()
            else:
                print(f'{Color.CYAN}out{Color.OFF}: {solve(parse_results)}')
        except ParseBaseException as err:
            print(f"{Color.RED}Parsing error!{Color.OFF}: {err}")
        except Exception as err:
            print(f"{Color.RED}Execution error!{Color.OFF}: {err}")
    except EOFError:
        sys.exit(0)
