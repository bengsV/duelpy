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
      There exists an order of the arms such that higher-ranked arms are (in expectation) preferable to lower-ranked arms.

    PAC
      Short for Probably Approximately Correct, an algorithm class which trades off correctness for lower runtime (or, in our case lower sample complexity). The result is guaranteed to be close to the correct one (by :math:`\epsilon`) with a given probability (:math:`\delta`). These algorithms usually require a certain amount of duels to come to a decision and are not designed to improve on it with further duels, if possible. Implementations of PAC algorithms in this library use an explore-then-exploit approach. This means that, when given a time horizon that exceeds the number of duels the algorithm needs to come to its decision, this decision is exploited by repeatedly dueling the determined best arm for the remaining duels.

    Low Noise Model
      No arms are tied and if an arm wins against another on average, its Borda score is higher.

    Strong Stochastic Transitivity
      For all pairwise distinct arms :math:`i,j,k \in [N]` such that :math:`\Delta_{i,j \geq 0}` and :math:`\Delta_{j,k} \geq 0`, it holds that :math:`\Delta_{i,k}\geq\max\{\Delta_{i,j},\Delta_{j,k}\}`. :math:`\Delta` is the :term:`calibrated preference matrix` here.

    Relaxed Stochastic Transitivity
      For :math:`\gamma \in (0,1)` and all pairwise distinct arms :math:`i,j,k \in [K]` such that :math:`\Delta_{i,j \geq 0}` and :math:`\Delta_{j,k}\geq 0`, it holds that :math:`\Delta_{i,k}\geq \gamma\max\{\Delta_{i,j},\Delta_{j,k}\}`. :math:`\Delta` is the :term:`calibrated preference matrix` here.

    Moderate Stochastic Transitivity
      For all pairwise distinct arms :math:`i,j,k \in [N]` such that :math:`\Delta_{i,j \geq 0}` and :math:`\Delta_{j,k} \geq 0`, it holds that :math:`\Delta_{i,k}\geq\min\{\Delta_{i,j},\Delta_{j,k}\}`. :math:`\Delta` is the :term:`calibrated preference matrix` here.

    Weak Stochastic Transitivity
      For any triplet of arms :math:`i,j,k \in [N]` with :math:`\Delta_{i,j\geq0}` and :math:`\Delta_{j,k}\geq0` implies :math:`\Delta_{i,k}\geq 0`. :math:`\Delta` is the :term:`calibrated preference matrix` here.

    Stochastic Triangle Inequality
      Given a total order of arms, for all arms :math:`i,j,k \in [N]` with :math:`\Delta_{i,j \geq 0}` and :math:`\Delta_{j,k}\geq 0` it holds that :math:`\Delta_{i,k}\leq\Delta_{i,j}+\Delta_{j,k}`. :math:`\Delta` is the :term:`calibrated preference matrix` here.

    Calibrated Preference Matrix
      We define a preference matrix :math:`P` with elements :math:`p_{i,j}` such that :math:`P(i \prec j)=p_{i,j}`. The calibrated preference matrix is defined as :math:`\Delta` with :math:`\Delta_{i,j}=p_{i,j}-0.5`. Sometimes it is useful for the definition of bounds for algorithms or other definitions to consider these differences instead of the actual probabilities.

    General Identifiability
      There exists an arm which has a higher probability of winning against any other arm than the arm that is the next best at beating that arm.

    Condorcet winner
      The unique arm which beats all others on average. It does not exist for all problems.

    Borda winner
      The arm with the best chance of beating a uniformly random chosen arm. It always exists and does not have to be equal to the :term:`Condorcet winner`, if one exists.

    Copeland winner
      The arm that beats most other arms on average. It always exists, but there may be more than one. If a :term:`Condorcet winner` exists, it is the only Copeland winner.

    Copeland score
      How many arms an arm beats on average. The normalized Copeland score is obtained by dividing the score by the number of arms minus one.

    Copeland regret
      Regret with respect to the :term:`Copeland winner`. Unless noted otherwise, the average Copeland regret is used. It is calculated by subtracting the average normalized :term:`Copeland score` of the dueled arms from the best :term:`Copeland score`.

    Copeland ranking
      A ranking of arms with respect to their :term:`Copeland score`.
    
    Mallows distribution
      Distribution for permutations. It is controlled by two parameters, a ground truth ranking and a spread parameter :math:`\phi`. This spread parameter controls how the probability of a ranking falls off the more it differs from the ground truth. We consider the case where the difference is measured using Kendall's :math:`\tau`. Alternatively Spearman's :math:`\rho` may be used (the spread parameter is then usually called :math:`\theta`). See :cite:`busa2014preference` for more details.
    
    Bradley-Terry distribution
      Similar to the :term:`Plackett-Luce distribution`.
    
    Plackett-Luce distribution
      The Plackett-Luce distribution assigns a utility :math:`u` to each arm. The utilities of two arms determine the probability of either arm winning against the other: :math:`P(i\prec j)=\frac{u_i}{u_i+u_j}`. For details on this distribution see :cite:`szorenyi2015online`.