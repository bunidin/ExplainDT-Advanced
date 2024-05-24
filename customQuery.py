import sys
import subprocess

def main(query_file, solver_path, model_path):
    # Read the query from the file
    with open(query_file, 'r') as f:
        query = f.read().strip()
    
    # Run the interpreter with the given query
    process = subprocess.Popen(
        ['python3', 'interpreter.py', solver_path, model_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Pass the query to the interpreter
    stdout, stderr = process.communicate(input=query)
    
    # Print the output
    print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python3 customQuery.py <query_file> <solver_path> <model_path>')
        sys.exit(1)
    
    query_file = sys.argv[1]
    solver_path = sys.argv[2]
    model_path = sys.argv[3]
    
    main(query_file, solver_path, model_path)
