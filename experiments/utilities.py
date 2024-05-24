import matplotlib.pyplot as plt
from os import system


def generate_trees(min_dim: int, max_dim: int) -> None:
    for dim in range(min_dim, max_dim + 1):
        call_str = "python3 dtrees/create_dtrees.py -o" \
            + f" d{dim}.json -d {dim}" \
            + f" -l {100*(dim-8)} -s 15000"
        system(call_str)


def plot_query_times(filename, query, sizes, times, results):
    _, ax = plt.subplots()
    sat_sizes = []
    not_sat_sizes = []
    sat_times = []
    not_sat_times = []

    for i in range(len(sizes)):
        if results[i] == 10:  # 10 equals sat
            sat_sizes.append(sizes[i])
            sat_times.append(times[i])
            continue
        not_sat_sizes.append(sizes[i])
        not_sat_times.append(times[i])

    ax.scatter(sat_sizes, sat_times, color='green')
    ax.scatter(not_sat_sizes, not_sat_times, color='red')
    ax.set_title("Size of tree vs time given the formula: " + query)
    plt.xlabel("Size (amount of nodes)")
    plt.ylabel("Time (in seconds)")
    plt.savefig(filename)
