########
Glossary
########

.. glossary::
    :sorted:

    Regret
      A metric which measures the quality of two chosen arms. We use three variants: strong, weak and average. Strong regret is the difference between the worst chosen arm and the best available arm. Weak regret uses the difference for the best chosen arm and average uses the mean of the two values. The regret is usually accumulated over many duels.
    
    Duel
      The process of obtaining preferential feedback for two chosen arms.
    
    Total Order
      ordering

    Probably Approximately Correct
      Algorith class which trades off correctness for lower runtime, the result is guaranteed to be close to the correct one (by :math:`\epsilon`) with a given probability (:math:`\delta`).

    low noise modes
      x

    Strong Stochastic Transitivity
      x

    Relaxed Stochastic Transitivity
      :math:`\gamma`

    Moderate Stochastic Transitivity
      x

    Weak Stochastic Transitivity
      x

    Stochastic Triangle Inequality
      x

    general identifiability
      x

    Condorcet winner
      The unique arm which beats all others on average. It does not exist for all problems.

    Borda winner
      The arm with the best chance of beating a uniformly random chosen arm. It always exists and must not be equal to the Condorcet winner, if one exists.

    von Neumann winner ?
      x

    Copeland winner
      The arm that beats most other arms on average. It always exists, but there may be more than one. If a Condorcet winner exists, it is the only Copeland winner.

    Copeland score
      How many arms an arm beats on average.

    Copeland regret
      Regret with respect to the Copeland winner.

    Random walk winner?
      x
