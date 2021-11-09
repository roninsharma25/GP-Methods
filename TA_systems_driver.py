# Author: Sam Champer

# This python utility file configures (via the command line) and
# runs the rat slim file, then parses, and prints the result.

from argparse import ArgumentParser
from slimutil import run_slim, configure_slim_command_line
from math import sqrt
from statistics import stdev


def parse_slim(slim_string, results):
    """
    Parse the output of SLiM and add the results to a dict.
    Args:
        slim_string: the entire output of a run of SLiM.
        results: a dict containing the desired output elements.
    """
    slim_lines = slim_string.split('\n')
    for line in slim_lines:
        gen_to_99 = 0
        end_rate = 0
        if line.startswith("GENS_TO_99:"):
            gen_to_99 = int(line.split("GENS_TO_99:")[1])
        if line.startswith("END_DRIVE_RATE:"):
            end_rate = float(line.split("END_DRIVE_RATE:")[1])

        # Increment results.
        results["Gens to 99 percent"] += gen_to_99
        results["Rate after 100 gens"] += end_rate
        results["Gens to 99 percent SE"].append(gen_to_99)
        results["Rate after 100 gens SE"].append(end_rate)


def main():
    """
    1. Configure using argparse.
    2. Generate cl string and run SLiM.
    3. Parse the output of SLiM.
    4. Print the results.
    """
    # Get args from arg parser:
    parser = ArgumentParser()
    parser.add_argument('-src', '--source', default="TA_systems.slim", type=str,
                        help="SLiM file to be run.")
    parser.add_argument('-header', '--print_header', action='store_true', default=False,
                        help='If this is set, python prints a header for a csv file.')
    parser.add_argument('-nreps', '--num_repeats', type=int, default=5,
                        help='Results will be averaged from this many simulations. Default 10.')

    # The following argument names exactly match the names of the variable parameters in SLiM.
    parser.add_argument('--DRIVE_FITNESS_VALUE', default=0.99, type=float)
    parser.add_argument('--DROP_RATE', default=0.25, type=float)
    parser.add_argument('--EMBRYO_RESISTANCE_RATE', default=0.95, type=float)
    parser.add_argument('--GERMLINE_RESISTANCE_RATE', default=0.99, type=float)

    args_dict = vars(parser.parse_args())
    sim_reps = args_dict.pop("num_repeats")

    if args_dict.pop("print_header", None):
        # Print the variable names. First bools, then the rest.
        # print(','.join(f"{arg}" for arg in args_dict if type(args_dict[arg]) == bool), end=",")  # No bool args in this driver.
        print(','.join(f"{arg}" for arg in args_dict if type(args_dict[arg]) != bool and arg != "source"), end=",")
        # Print headings for the data being collected:
        print("Gens to 99 percent," \
            "Gens to 99 percent SE," \
            "Rate after 100 gens," \
            "Rate after 100 gens SE")

    # Assemble the command line arguments to use for SLiM:
    clargs = configure_slim_command_line(args_dict)

    results = {"Gens to 99 percent": 0, "Gens to 99 percent SE": [], "Rate after 100 gens": 0, "Rate after 100 gens SE": []}

    for _ in range(sim_reps):
        # Run the file with the desired arguments.
        slim_output = run_slim(clargs)

        # Parse and analyze the result, updating the results dict.
        parse_slim(slim_output, results)

    # Calculate SEs. The multiply by sim_reps is because all values in "results" are divided by sim_reps when output below.
    if sim_reps > 1:
        results["Rate after 100 gens SE"] = stdev(results["Rate after 100 gens SE"]) / sqrt(sim_reps) * sim_reps
        results["Gens to 99 percent SE"] = stdev(results["Gens to 99 percent SE"]) / sqrt(sim_reps) * sim_reps

    # Print the results, starting with the parameters.
    # print(','.join(f"{args_dict[arg]}" for arg in args_dict if type(args_dict[arg]) == bool), end=",")  # No bool args in this driver.
    print(','.join(f"{args_dict[arg]}" for arg in args_dict if type(args_dict[arg]) != bool), end=",")
    print(','.join(f"{i / sim_reps}" for i in results.values()))


if __name__ == "__main__":
    main()
