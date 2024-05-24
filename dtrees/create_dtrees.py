import argparse
import dtree
import pathlib

parser = argparse.ArgumentParser(description="Generator of Decision Trees")

# So far only supports random DTs (i.e., trained on randomly generated data)

parser.add_argument('-o', '--output', help='Output json file for the DT', required=True)
parser.add_argument('-d', '--dim', help='Dimension of the resulting DT', type=int, required=True)
parser.add_argument('-l', '--leaves', help='Max number of leaves', type=int, required=True)
parser.add_argument('-s', '--samples', help='Number of samples to train on', type=int, required=True)
parser.add_argument('-v', '--verbose', help='Verbose mode', action='store_true')

args = parser.parse_args()

base_path = pathlib.Path(__file__).parent.absolute() / "generated_trees/"
output_file = base_path / args.output
dim = args.dim
n_leaves = args.leaves
n_training_samples = args.samples
verbose = args.verbose

if verbose:
    print(vars(args))

dtree.random_dt_to_json(output_file, dim, n_leaves, n_training_samples)

if verbose:
    print(f"Decision Tree generated and serialized into {output_file}")
