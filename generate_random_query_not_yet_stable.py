import random

# Define possible values and operations
constants = ['true', 'false', '0', '1', '?']
operations = ['and', 'or', 'not', 'implies']
quantifiers = ['exists', 'for all']
unary_predicates = ['full', 'all_pos', 'all_neg', 'RFS', 'node']
binary_predicates = ['subsumed by', 'is subsumed by', '<=', 'cons', 'LEL', 'SR']

def random_varname():
    return f'var{random.randint(1, 100)}'

def random_constant():
    return random.choice(constants)

def random_unary_predicate():
    pred = random.choice(unary_predicates)
    var_or_const = random.choice([random_varname(), random_constant()])
    return f'{pred}({var_or_const})'

def random_binary_predicate():
    pred = random.choice(binary_predicates)
    left = random.choice([random_varname(), random_constant()])
    right = random.choice([random_varname(), random_constant()])
    return f'{left} {pred} {right}'

def random_classification():
    class_name = f'class{random.randint(1, 10)}'
    var_or_const = random.choice([random_varname(), random_constant()])
    return f'{class_name}({var_or_const})'

def random_relevant():
    feature = f'feature{random.randint(1, 20)}'
    return f'relevant {feature}'

def random_atomic():
    choices = [
        random_unary_predicate,
        random_binary_predicate,
        random_classification,
        random_relevant
    ]
    return random.choice(choices)()

def random_quantified_sentence(max_depth=3):
    if max_depth <= 0:
        return random_atomic()
    quantifier = random.choice(quantifiers)
    var = random_varname()
    subquery = random_query(max_depth - 1)
    return f'{quantifier} {var}, {subquery}'

def random_query(max_depth=3):
    if max_depth <= 0:
        return random_atomic()
    operation = random.choice(operations)
    if operation == 'not':
        subquery = random_query(max_depth - 1)
        return f'not ({subquery})'
    left = random_query(max_depth - 1)
    right = random_query(max_depth - 1)
    return f'({left}) {operation} ({right})'

def generate_random_query():
    return random.choice([
        random_atomic,
        lambda: random_quantified_sentence(3),
        lambda: random_query(3)
    ])()

if __name__ == "__main__":
    random_query = generate_random_query()
    print(f'Random Query: {random_query}')
