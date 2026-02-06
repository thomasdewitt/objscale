# objscale/__init__.py
from ._size_distributions import (
    finite_array_size_distribution,
    finite_array_powerlaw_exponent,
    array_size_distribution,
)

from ._fractal_dimensions import (
    individual_fractal_dimension,
    ensemble_correlation_dimension,
    ensemble_box_dimension,
    total_perimeter,
    total_number,
    isolate_largest_structure,
    coarsen_array,
    get_coords_of_boundaries,
    get_locations_from_pixel_sizes,
    correlation_integral,
    label_size,
)

from ._object_analysis import (
    get_structure_props,
    get_every_boundary_perimeter,
    remove_structures_touching_border_nan,
    label_periodic_boundaries,
    remove_structure_holes,
    clear_border_adjacent,
)

from ._utils import (
    linear_regression,
    encase_in_value,
)

__all__ = [
    'finite_array_size_distribution',
    'finite_array_powerlaw_exponent',
    'array_size_distribution',
    'individual_fractal_dimension',
    'ensemble_correlation_dimension',
    'ensemble_box_dimension',
    'total_perimeter',
    'total_number',
    'isolate_largest_structure',
    'coarsen_array',
    'get_coords_of_boundaries',
    'get_locations_from_pixel_sizes',
    'correlation_integral',
    'label_size',
    'get_structure_props',
    'get_every_boundary_perimeter',
    'remove_structures_touching_border_nan',
    'label_periodic_boundaries',
    'remove_structure_holes',
    'clear_border_adjacent',
    'linear_regression',
    'encase_in_value',
]

__version__ = "0.2.1"
__author__ = "Thomas DeWitt"
__email__ = "thomas.dewitt@utah.edu"
__description__ = "Object-based analysis functions for fractal dimensions and size distributions"