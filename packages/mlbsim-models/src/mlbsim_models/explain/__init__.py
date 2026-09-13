"""Explainability: per-prediction factors + natural-language rendering."""

from mlbsim_models.explain.factors import linear_shap, top_factors
from mlbsim_models.explain.nl_explainer import build_prompt, render_explanation

__all__ = ["build_prompt", "linear_shap", "render_explanation", "top_factors"]
