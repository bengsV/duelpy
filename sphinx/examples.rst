########
Examples
########

In the following, we list some examples of how the duelpy library can be used.
The first one shows how algorithms can be evaluated in experiments. The second one how a new algorithm might be used in an application and the third how a new algorithm can be implemented.

*******************
Running experiments
*******************
The `cli.py` program from the experiments package can be used to compare the performance of different algorithms.

An example usage might look like this:

.. code-block:: Bash

    python3 -m duelpy.experiments.cli --arms 10 --time-horizon 10000 --runs 10 --environment PlackettLuceModel --algorithms PlackettLucePACItem MallowsMPI

Here the :class:`PlackettLucePACItem<duelpy.algorithms.plackett_luce.PlackettLucePACItem>` and the :class:`MallowsMPI<duelpy.algorithms.mallows.MallowsMPI>` algorithms are compared using a :class:`PlackettLuceModel<duelpy.experiments.environments.PlackettLuceModel>` environment.
The time horizon is set to :math:`10000` and the experiment is repeated ten times to get averaged results.
For an overview of all parameterization options, try:

.. code-block:: Bash

    python3 -m duelpy.experiments.cli --help

***********
Application
***********
This is a toy example demonstrating how an algorithm could be used in an application.
A user is supposed to find the best among a set of options. Since this is usually a hard problem for a human to solve, we instead ask for pairwise preferences.
The results of these queries are then used to find the best option based on the users preferences.


 .. code-block:: python

    """A simple example of how a decision-tool using duelpy could work."""

    from duelpy.algorithms.savage import Savage
    from duelpy.feedback import CommandlineFeedback


    def _run_decision_experiment() -> None:
        arms = [
            "The best arm.",  # 0
            "The second best arm.",  # 1
            "The other second best arm, flip a coin.",  # 2
            "Third best arm.",  # 3
            "Least favorite arm.",  # 4
        ]
        feedback_mechanism = CommandlineFeedback(arms)
        algorithm = Savage(feedback_mechanism=feedback_mechanism, failure_probability=0.5)
        while not algorithm.is_finished():
            algorithm.step()
            print("Preference estimate is now")
            print(algorithm.preference_estimate)
        print("Estimated Copeland winner:")
        print(algorithm.get_copeland_winner())


    if __name__ == "__main__":
        _run_decision_experiment()

***********************
Writing a new algorithm
***********************
Algorithms should inherit from the :class:`Algorithm<duelpy.algorithms.algorithm.Algorithm>` class. This class defines a general structure, consisting of ``step``, ``is_finished``, and ``run`` functions.
Also, the constructor takes two parameters and sets the corresponding attributes ``feedback_mechanism`` and ``time_horizon``.
The feedback mechanism models the environment, storing how many arms are available and providing the ``duel`` function to compare two arms.
The :class:`FeedbackMechanism<duelpy.feedback.FeedbackMechanism>` module contains some implementations. If necessary, a new implementation for a specific application is possible by extending the ``FeedbackMechanism`` class.
The time horizon is an upper bound on the duels (comparisons) to be made by the algorithm. The value ``None`` is interpreted as an infinite time horizon. The programmer is responsible for keeping this limit.
If this complicates the implementation and it is strictly necessary to not exceed the time horizon, the ``BudgetedFeedbackMechanism`` may help.
The basic structure of an algorithm is defined by its ``step`` function. What is done in one step can be decided by the designer. Some possible options are dueling two arms once per call or executing one logical step.
For convenience, a ``run`` function is implemented which calls ``step`` until ``is_finished`` is true. For most algorithms, only ``step`` and often ``is_finished`` need to be overridden in the subclasses.

The ``Algorithm`` class has extensions, which define additional functions for returning the results.
Possible targets are finding a Copeland winner or creating a ranking.
The interfaces are located in the :class:`interfaces<duelpy.algorithms.interfaces>` module.
Some interfaces include:

- :class:`CondorcetProducer<duelpy.algorithms.interfaces.CondorcetProducer>`
- :class:`SingleCopelandProducer<duelpy.algorithms.interfaces.SingleCopelandProducer>`
- :class:`AllCopelandProducer<duelpy.algorithms.interfaces.AllCopelandProducer>`
- :class:`PartialRankingProducer<duelpy.algorithms.interfaces.PartialRankingProducer>`
- ...

If there exists no interface which fits the algorithm, the superclass :class:`Algorithm<duelpy.algorithms.algorithm.Algorithm>` may be used instead.

An example of the structure is given here:
 .. code-block:: python

    from duelpy.algorithms.interfaces import CondorcetProducer

    class MyAlgorithm(CondorcetProducer):
        def __init__(self, feedback_mechanism, time_horizon, ...):
            super().__init__(feedback_mechanism, time_horizon)
            # self.feedback_mechanism and self.time_horizon are now defined
            # initialization
        
        def step(self):
            # execute one step
        
        def is_finished(self):
            # determine if the algorithm has terminated
            return finished

 
The parameters in the constructor are ordered as follows. First, there are the standard parameters
``feedback_mechanism``  and ``time horizon``. Then algorithm-specific parameters, these can include a
``random_state``, a ``np.random.RandomState`` object used for randomization, or ``epsilon`` and
``failure_probability`` for (:math:`\epsilon`, :math:`\delta`)-PAC algorithms.

It may be helpful to store statistics about the duels in a :doc:`autoapi/duelpy/stats/preference_estimate/index` object, which allows for calculating mean and confidence values.
The confidence radius can be freely defined, mostly a ``HoeffdingConfidenceRadius`` from the :doc:`autoapi/duelpy/stats/confidence_radius/index` module is used.

The new implementation can be benchmarked using the :doc:`autoapi/duelpy/experiments/index` package.
