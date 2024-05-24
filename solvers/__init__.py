from subprocess import run, PIPE

from context import Context


def run_solver(solver_path: str, file_path: str):
    # TODO: Add any interpretation of results here?
    # return run([solver_path, file_path], stdout=PIPE).stdout
    return run([solver_path, file_path], stdout=PIPE)


def extract_meaning(solver_std_out: bytes, context: Context) -> dict:
    vars = []
    var_lines = [
        line[2:].rstrip('\n')
        for line in str(solver_std_out).split("\\n") if line[0] == 'v'
    ]
    for line in var_lines:
        vars.extend([var for var in map(lambda x: int(x), line.split(' '))])
    reverse_dict = {
        value: key
        for key, value in context.V.items()
    }
    return {
        reverse_dict[abs(var)]: var > 0
        for var in vars if abs(var) in reverse_dict.keys()
    }
