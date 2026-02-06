"""
防退化护栏模块
"""
from .prior_anchor import (
    compute_prior_anchor_loss,
    compute_prior_anchor_loss_multi_view,
)
from .smoothness import (
    compute_spherical_smoothness_loss,
    compute_spherical_smoothness_loss_multi_view,
)
from .scale_guard import compute_scale_constraint_loss
from .far_field_mask import (
    create_far_field_mask,
    create_far_field_masks,
    apply_far_field_mask_to_loss,
)
from .bspline_constraints import (
    compute_monotonicity_loss,
    compute_directional_smoothness_loss,
    compute_far_field_asymptotic_loss,
    compute_bspline_constraints_loss,
    compute_bspline_constraints_loss_multi_view,
)
from .regularization_loss import RegularizationLoss

__all__ = [
    'compute_prior_anchor_loss',
    'compute_prior_anchor_loss_multi_view',
    'compute_spherical_smoothness_loss',
    'compute_spherical_smoothness_loss_multi_view',
    'compute_scale_constraint_loss',
    'create_far_field_mask',
    'create_far_field_masks',
    'apply_far_field_mask_to_loss',
    'compute_monotonicity_loss',
    'compute_directional_smoothness_loss',
    'compute_far_field_asymptotic_loss',
    'compute_bspline_constraints_loss',
    'compute_bspline_constraints_loss_multi_view',
    'RegularizationLoss',
]
