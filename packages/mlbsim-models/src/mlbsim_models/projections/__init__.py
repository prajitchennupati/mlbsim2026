"""Player talent projections.

``marcel``     Tango's Marcel: 5/4/3-weighted prior years, regression to the mean,
               age curve. The honest baseline every fancier model is measured against.
``shrinkage``  empirical-Bayes partial pooling — the lightweight "hierarchical"
               upgrade; a full PyMC model is a documented future item.
"""

from mlbsim_models.projections.marcel import MarcelProjection, project_marcel
from mlbsim_models.projections.shrinkage import shrink_rates

__all__ = ["MarcelProjection", "project_marcel", "shrink_rates"]
