"""Loom: a looped-MoE token decoder with concept-level guidance in its own
latent space. Token level stays autoregressive (dense CE — the lesson from
the Levencode stage-5 postmortem); the concept level only steers."""

from .concept import ConceptPredictor, concept_loss, pool_segments, shift_concepts
from .config import LoomConfig
from .model import LoomLM, param_report
