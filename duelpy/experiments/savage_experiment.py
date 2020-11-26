"""Reproduce the experiments from the SAVAGE paper.

Implementation of parts of the experiments in section 5 of
:cite:`urvoy2013generic`. Currently only one problem setting ("Hard Condorcet
Matrices") is implemented, so this alone is not necessarily representative of
Algorithm performance.
"""

import argparse
import inspect
import time
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Type

from joblib import delayed
from joblib import Parallel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from duelpy.algorithms import Algorithm
from duelpy.algorithms import algorithm_list
from duelpy.experiments.environments import HardCondorcetMatrix


def run_single_algorithm(
    task_random_state: np.random.RandomState,
    num_arms: int,
    algorithm_class: Type[Algorithm],
    parameters: Dict,
    run_id: int,
) -> pd.DataFrame:
    """Execute one algorithm for one problem setting and return the results."""
    # This function has to juggle with a lot of data and put it into a shape
    # that is convenient for plotting. In the future it might be a good idea to
    # move this into a stats module or into the FeedbackMechanism itself, but
    # for now we need all those locals.
    # pylint: disable=too-many-locals
    data: Dict[str, Any] = {
        "algorithm": [],
        "run_id": [],
        "time_step": [],
        "wall_clock": [],
        "cum_average_regret": [],
    }

    feedback_mechanism = HardCondorcetMatrix(
        num_arms=num_arms, random_state=task_random_state
    )
    # Filter accepted parameters.
    parameters["random_state"] = task_random_state
    parameters_to_pass = dict()
    for parameter in parameters.keys():
        if parameter in inspect.getfullargspec(algorithm_class.__init__)[0]:
            parameters_to_pass[parameter] = parameters[parameter]
    algorithm = algorithm_class(feedback_mechanism, **parameters_to_pass)
    best = feedback_mechanism.preference_matrix.get_condorcet_winner()
    assert best is not None
    start_time = time.time()
    elapsed = dict()
    while not algorithm.is_finished():
        algorithm.step()
        elapsed[feedback_mechanism.get_num_duels()] = time.time() - start_time
    (regret_history, _) = feedback_mechanism.calculate_average_regret(best_arm=best)
    cumulative_regret = 0.0
    data["algorithm"].append(algorithm_class.__name__)
    data["run_id"].append(run_id)
    data["time_step"].append(0)
    data["cum_average_regret"].append(0)
    data["wall_clock"].append(0)
    for (time_step, regret_value) in enumerate(regret_history, start=1):
        cumulative_regret += regret_value
        data["algorithm"].append(algorithm_class.__name__)
        data["run_id"].append(run_id)
        data["time_step"].append(time_step)
        data["cum_average_regret"].append(cumulative_regret)
        data["wall_clock"].append(elapsed[time_step] if time_step in elapsed else None)
    return pd.DataFrame(data)


# This function will be replaced in order to make the experiments more
# flexible, so its not worth refactoring this right now.
# pylint: disable=too-many-arguments
def run_experiment(
    algorithms: List[Type[Algorithm]],
    time_horizon: int,
    num_arms: int,
    runs: int,
    n_jobs: int,
    base_random_seed: int,
) -> pd.DataFrame:
    """Run the experiment.

    Parameters
    ----------
    algorithms
        A list of algorithms to run.
    time_horizon
        For how long each algorithm should be run.
    num_arms
        How many arms to include in the generated problems.
    runs
        How often each algorithm should be run.
    n_jobs
        How many jobs to execute in parallel.
    base_random_seed
        Used to derive random states for each experiment.

    Returns
    -------
    DataFrame
        The experiment results.
    """
    start_time = time.time()
    failure_probability = (
        1 / time_horizon
    )  # for PAC algorithms, to give them some time for exploitation as well

    parameters = {
        "time_horizon": time_horizon,
        "failure_probability": failure_probability,
    }

    def job_producer() -> Generator:
        for algorithm_class in algorithms:
            algorithm_name = algorithm_class.__name__
            for run_id in range(runs):
                random_state = np.random.RandomState(
                    (base_random_seed + hash(algorithm_name) + run_id) % 2 ** 32
                )
                yield delayed(run_single_algorithm)(
                    random_state, num_arms, algorithm_class, parameters, run_id,
                )

    jobs = list(job_producer())
    result = Parallel(n_jobs=n_jobs, verbose=10)(jobs)
    runtime = time.time() - start_time
    print(f"Experiments took {round(runtime)}s.")
    return pd.concat(result)


def plot_results(data: pd.DataFrame) -> None:
    """Plot experiment results using seaborn.

    Parameters
    ----------
    data
        A pandas dataframe with columns time_step, cum_average_regret and
        wall_clock.
    """
    sns.set()
    _fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    sns.lineplot(
        data=data,
        x="time_step",
        y="cum_average_regret",
        hue="algorithm",
        ci=None,
        linewidth=2,
        ax=ax1,
    )
    sns.lineplot(
        data=data,
        x="time_step",
        y="wall_clock",
        hue="algorithm",
        ci=None,
        linewidth=2,
        ax=ax2,
    )
    plt.show()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PB-MAB experiments and plot results."
    )
    algorithm_names_to_algorithms = {
        algorithm.__name__: algorithm for algorithm in algorithm_list
    }
    algorithm_choices_string = " ".join(algorithm_names_to_algorithms.keys())
    parser.add_argument(
        "-a, --algorithms",
        nargs="+",
        metavar="ClassName",
        dest="algorithms",
        default=algorithm_names_to_algorithms.keys(),
        help=f"Algorithms to compare. (default: {algorithm_choices_string})",
        choices=algorithm_names_to_algorithms.keys(),
    )
    parser.add_argument(
        "--runs",
        dest="runs",
        default=10,  # savage paper uses 1000
        type=int,
        help="How often to run each algorithm. Results will be averaged. (default: 10)",
    )
    parser.add_argument(
        "--arms",
        dest="num_arms",
        default=20,  # savage paper uses 100, but our current savage implementation does not scale well
        type=int,
        help="How many arms the generated preference matrices should contain. (default: 20)",
    )
    parser.add_argument(
        "--time-horizon",
        dest="time_horizon",
        default=int(1e4),  # savage paper uses 1e6
        type=int,
        help="For how many time steps to run each algorithm. (default: 1e4)",
    )
    parser.add_argument(
        "--random-seed",
        dest="base_random_seed",
        default=42,
        type=int,
        help="Base random seed for reproducible results.",
    )
    parser.add_argument(
        "--jobs",
        dest="n_jobs",
        default=-1,
        type=int,
        help="How many experiments to run in parallel. The special value -1 stands for the number of processor cores.",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        default=False,
        help="Enable profiling mode. This disables parallelism and plotting. Intended for use with cProfile.",
    )

    args = parser.parse_args()
    algorithms = [
        algorithm_names_to_algorithms[algorithm] for algorithm in args.algorithms
    ]

    results = run_experiment(
        algorithms=algorithms,
        n_jobs=1 if args.profile else args.n_jobs,
        time_horizon=args.time_horizon,
        num_arms=args.num_arms,
        runs=args.runs,
        base_random_seed=args.base_random_seed,
    )
    if not args.profile:
        plot_results(results)


if __name__ == "__main__":
    _main()
