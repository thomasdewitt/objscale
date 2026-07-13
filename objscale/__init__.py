"""
Object-based analysis of 2D binary arrays: fractal dimensions, size
distributions, and object scaling properties, with corrections for
finite-domain and resolution effects. Optimized for large datasets.

If you use this package, cite both DeWitt & Garrett (2024),
https://doi.org/10.5194/acp-24-8457-2024, and DeWitt et al. (2026),
https://doi.org/10.5194/acp-26-6951-2026.

AI agents: a complete usage skill for this package is bundled with the
installation. Before using objscale, install it with
``objscale.install_agent_skill('claude')`` (or ``'codex'``), or read it
directly at the path returned by ``objscale.skill_path()``. For other
agent frameworks, copy the file at ``objscale.skill_path()`` into your
skills directory manually.
"""
from ._size_distributions import (
    finite_array_size_distribution,
    finite_array_powerlaw_exponent,
    array_size_distribution,
)

from ._fractal_dimensions import (
    individual_fractal_dimension,
    individual_correlation_dimension,
    ensemble_correlation_dimension,
    ensemble_box_dimension,
    ensemble_information_dimension,
    ensemble_box_renyi_dimension,
    ensemble_sandbox_renyi_dimension,
    total_perimeter,
    total_number,
    isolate_nth_largest_structure,
    coarsen_array,
    get_coords_of_boundaries,
    get_locations_from_pixel_sizes,
    label_size,
)

from ._object_analysis import (
    label_structures,
    get_structure_props,
    get_structure_areas,
    get_structure_perimeters,
    get_structure_height_width,
    get_every_boundary_perimeter,
    remove_structures_touching_border_nan,
    remove_structure_holes,
    clear_border_adjacent,
)

from ._utils import (
    linear_regression,
    encase_in_value,
    set_num_threads,
)

from ._skill import skill_path, install_agent_skill

__all__ = [
    'finite_array_size_distribution',
    'finite_array_powerlaw_exponent',
    'array_size_distribution',
    'individual_fractal_dimension',
    'individual_correlation_dimension',
    'ensemble_correlation_dimension',
    'ensemble_box_dimension',
    'ensemble_information_dimension',
    'ensemble_box_renyi_dimension',
    'ensemble_sandbox_renyi_dimension',
    'total_perimeter',
    'total_number',
    'isolate_nth_largest_structure',
    'coarsen_array',
    'get_coords_of_boundaries',
    'get_locations_from_pixel_sizes',
    'label_size',
    'get_structure_props',
    'get_structure_areas',
    'get_structure_perimeters',
    'get_structure_height_width',
    'get_every_boundary_perimeter',
    'remove_structures_touching_border_nan',
    'label_structures',
    'remove_structure_holes',
    'clear_border_adjacent',
    'linear_regression',
    'encase_in_value',
    'set_num_threads',
    'skill_path',
    'install_agent_skill',
]

__version__ = "2.0.0"
__author__ = "Thomas DeWitt"
__email__ = "thomas.dewitt@utah.edu"
__description__ = "Object-based analysis functions for fractal dimensions and size distributions"