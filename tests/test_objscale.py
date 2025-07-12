#!/usr/bin/env python3
"""
Test script for objscale package
Creates 4 percolation lattices and calculates all parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.path.insert(0, '..')
import objscale

# Set random seed for reproducibility
np.random.seed(42)

# Percolation threshold for 2D square lattice
P_C = 0.5927

def create_percolation_lattice(size, p=P_C):
    """Create a percolation lattice with given probability"""
    return (np.random.random((size, size)) < p).astype(int)

def main():
    print("=== OBJSCALE PACKAGE TEST ===")
    size = 1000
    print(f"Creating 4 percolation lattices ({size}x{size}) with p = {P_C}")
    
    # Create 4 percolation lattices
    arrays = []
    for i in range(4):
        print(f"Creating array {i+1}/4...")
        array = create_percolation_lattice(size, P_C)
        arrays.append(array)
        print(f"  Array {i+1}: {np.sum(array)} occupied sites")
    
    print("\n=== CALCULATING PARAMETERS ===")
    
    # Calculate power-law exponents and get distribution data
    s = time.time()
    print("Calculating area power-law exponent and distribution...")
    (area_exponent, area_error), (area_sizes, area_counts) = objscale.finite_array_powerlaw_exponent(
        arrays, 'area', bins=50, min_threshold=10, return_counts=True
    )
    
    print("Calculating perimeter power-law exponent and distribution...")
    (perim_exponent, perim_error), (perim_sizes, perim_counts) = objscale.finite_array_powerlaw_exponent(
        arrays, 'perimeter', bins=50, min_threshold=10, return_counts=True
    )
    
    print(f'Power law exponents took {time.time()-s:.02f} seconds')
    print(f"    Area exponent: {area_exponent:.3f} ± {area_error:.3f}")
    print(f"    Perimeter exponent: {perim_exponent:.3f} ± {perim_error:.3f}")
    
    # Calculate correlation dimension
    print("Calculating correlation dimension...")
    corr_dim, corr_error, corr_lengths, corr_integrals = objscale.ensemble_correlation_dimension(
        arrays, return_C_l=True, point_reduction_factor=1000, maxlength='auto', interior_circles_only=False
    )
    print(f"Correlation dimension: {corr_dim:.3f} ± {corr_error:.3f}")
    
    # Calculate box dimension
    print("Calculating box dimension...")
    box_dim, box_error, box_sizes, box_counts = objscale.ensemble_box_dimension(
        arrays, return_values=True
    )
    print(f"Box dimension: {box_dim:.3f} ± {box_error:.3f}")
    
    # Calculate individual fractal dimensions
    print("Calculating individual fractal dimensions...")
    individual_dims = []
    for i, array in enumerate(arrays):
        dim, error = objscale.individual_fractal_dimension([array])
        individual_dims.append((dim, error))
        print(f"  Array {i+1}: {dim:.3f} ± {error:.3f}")
    
    # Calculate coarsening dimension
    print("Calculating coarsening dimension...")
    coarsening_dim, coarsening_error, coarsening_factors, mean_total_perimeters = objscale.ensemble_coarsening_dimension(
        arrays, return_values=True, min_pixels = 3
    )
    print(f"Coarsening dimension: {coarsening_dim:.3f} ± {coarsening_error:.3f}")
    
    print("\n=== TESTING ADDITIONAL FUNCTIONS ===")
    
    # Test total_number
    print("Testing total_number...")
    test_array = arrays[0]
    num_structures = objscale.total_number(test_array)
    print(f"Total number of structures: {num_structures}")
    
    # Test total_perimeter
    print("Testing total_perimeter...")
    x_sizes = np.ones(test_array.shape)
    y_sizes = np.ones(test_array.shape)
    total_perim = objscale.total_perimeter(test_array.astype(np.float32), x_sizes, y_sizes)
    print(f"Total perimeter: {total_perim:.2f}")
    
    # Test isolate_largest_structure
    print("Testing isolate_largest_structure...")
    largest_struct = objscale.isolate_largest_structure(test_array)
    print(f"Largest structure has {np.sum(largest_struct)} pixels")
    
    # Test coarsen_array
    print("Testing coarsen_array...")
    coarsened = objscale.coarsen_array(test_array, 4)
    print(f"Original array shape: {test_array.shape}, Coarsened array shape: {coarsened.shape}")
    
    # Test get_structure_props
    print("Testing get_structure_props...")
    try:
        perims, areas, widths, heights = objscale.get_structure_props(test_array, x_sizes, y_sizes)
        print(f"Structure properties calculated: {len(perims)} structures found")
        if len(perims) > 0:
            print(f"  Average perimeter: {np.mean(perims):.2f}")
            print(f"  Average area: {np.mean(areas):.2f}")
    except Exception as e:
        print(f"  get_structure_props: {e}")
    
    # Test remove_structures_touching_border_nan
    print("Testing remove_structures_touching_border_nan...")
    # Create a test array with NaN borders
    test_array_nan = objscale.encase_in_value(test_array.astype(np.float32), np.nan)
    cleared_array = objscale.remove_structures_touching_border_nan(test_array_nan)
    print(f"  Structures after border removal: {np.sum(cleared_array == 1)}")
    
    # Test remove_structure_holes
    print("Testing remove_structure_holes...")
    filled_array = objscale.remove_structure_holes(test_array.astype(np.float32))
    print(f"  Structures after hole filling: {np.sum(filled_array == 1)}")
    
    # Test clear_border_adjacent
    print("Testing clear_border_adjacent...")
    cleared_border = objscale.clear_border_adjacent(test_array)
    print(f"  Structures after border clearing: {np.sum(cleared_border)}")
    
    # Test linear_regression
    print("Testing linear_regression...")
    x_test = np.array([1, 2, 3, 4, 5])
    y_test = np.array([2, 4, 6, 8, 10])
    (slope, intercept), (slope_err, intercept_err) = objscale.linear_regression(x_test, y_test)
    print(f"  Linear regression: slope={slope:.3f}±{slope_err:.3f}, intercept={intercept:.3f}±{intercept_err:.3f}")
    
    # Test encase_in_value
    print("Testing encase_in_value...")
    small_array = np.ones((3, 3))
    encased = objscale.encase_in_value(small_array, 0)
    print(f"  Original shape: {small_array.shape}, Encased shape: {encased.shape}")
    
    # Test size distribution functions
    print("Testing size distribution functions...")
    try:
        bin_middles, nontruncated_counts, truncated_counts, truncation_index = objscale.finite_array_size_distribution(arrays, 'area')
        print(f"  Finite array size distribution: {len(bin_middles)} size bins")
        print(f"    Truncation index: {truncation_index}")
        print(f"    Total structures: {np.sum(nontruncated_counts + truncated_counts)}")
        
        sizes_single, counts_single = objscale.array_size_distribution(test_array, 'area')
        print(f"  Single array size distribution: {len(sizes_single)} size bins")
        print(f"    Total structures: {np.sum(counts_single)}")
        
        print("  All additional function tests completed successfully!")
    except Exception as e:
        print(f"  Size distribution functions: {e}")
    
    print("\n=== CREATING PLOTS ===")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Objscale Package Test Results', fontsize=16)
    
    # Plot area distribution (convert from log10 back to linear for plotting)
    ax1 = axes[0, 0]
    ax1.loglog(10**area_sizes, 10**area_counts, 'bo-', markersize=4, linewidth=1)
    ax1.set_xlabel('Area')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Area Distribution\\nExponent: {area_exponent:.3f} ± {area_error:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # Plot perimeter distribution (convert from log10 back to linear for plotting)
    ax2 = axes[0, 1]
    ax2.loglog(10**perim_sizes, 10**perim_counts, 'ro-', markersize=4, linewidth=1)
    ax2.set_xlabel('Perimeter')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Perimeter Distribution\\nExponent: {perim_exponent:.3f} ± {perim_error:.3f}')
    ax2.grid(True, alpha=0.3)
    
    # Plot correlation integral
    ax3 = axes[1, 0]
    ax3.loglog(corr_lengths, corr_integrals, 'go-', markersize=4, linewidth=1)
    ax3.set_xlabel('Length Scale')
    ax3.set_ylabel('Correlation Integral')
    ax3.set_title(f'Correlation Integral\\nDimension: {corr_dim:.3f} ± {corr_error:.3f}')
    ax3.grid(True, alpha=0.3)
    
    # Plot box dimension
    ax4 = axes[1, 1]
    ax4.loglog(box_sizes, box_counts, 'mo-', markersize=4, linewidth=1)
    ax4.set_xlabel('Box Size')
    ax4.set_ylabel('Number of Boxes')
    ax4.set_title(f'Box Dimension\\nDimension: {box_dim:.3f} ± {box_error:.3f}')
    ax4.grid(True, alpha=0.3)
    
    # Plot coarsening dimension (total perimeter vs resolution)
    ax5 = axes[1, 2]
    ax5.loglog(coarsening_factors, mean_total_perimeters, 'co-', markersize=4, linewidth=1)
    ax5.set_xlabel('Resolution (coarsening factor)')
    ax5.set_ylabel('Total Perimeter')
    ax5.set_title(f'Coarsening Dimension\\nDimension: {coarsening_dim:.3f} ± {coarsening_error:.3f}')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== SUMMARY ===")
    print(f"Area exponent: {area_exponent:.3f} ± {area_error:.3f}")
    print(f"Perimeter exponent: {perim_exponent:.3f} ± {perim_error:.3f}")
    print(f"Correlation dimension: {corr_dim:.3f} ± {corr_error:.3f}")
    print(f"Box dimension: {box_dim:.3f} ± {box_error:.3f}")
    print(f"Coarsening dimension: {coarsening_dim:.3f} ± {coarsening_error:.3f}")
    print("Individual fractal dimensions:")
    for i, (dim, error) in enumerate(individual_dims):
        print(f"  Array {i+1}: {dim:.3f} ± {error:.3f}")
    
    print("\n=== TEST COMPLETED ===")

if __name__ == "__main__":
    main()