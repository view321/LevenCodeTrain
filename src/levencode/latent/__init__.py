"""Multi-granularity latent JEPA stack: teacher-latent distillation with
discrete plan anchors + continuous residuals, CALM-robust decodability,
and closed-loop latent-guided sampling."""

from .bundle import LatentBundle
from .chunker import HierarchicalSpec, LevelSpec, hierarchical_spans
from .rvq import RVQ
from .teacher import PrecomputedLatents, TeacherExtractor, load_teacher
