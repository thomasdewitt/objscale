#!/usr/bin/env python3
"""
Test script for objscale package
Creates 4 percolation lattices and calculates all parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
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
    print(f"Creating 4 percolation lattices (1000x1000) with p = {P_C}")
    
    # Create 4 percolation lattices
    arrays = []
    for i in range(4):
        print(f"Creating array {i+1}/4...")
        array = create_percolation_lattice(2000, P_C)
        arrays.append(array)
        print(f"  Array {i+1}: {np.sum(array)} occupied sites")
    
    print("\n=== CALCULATING PARAMETERS ===")
    
    # Calculate power-law exponents and get distribution data
    print("Calculating area power-law exponent and distribution...")
    (area_exponent, area_error), (area_sizes, area_counts) = objscale.finite_array_powerlaw_exponent(
        arrays, 'area', bins=50, min_threshold=10, return_counts=True
    )
    
    print("Calculating perimeter power-law exponent and distribution...")
    (perim_exponent, perim_error), (perim_sizes, perim_counts) = objscale.finite_array_powerlaw_exponent(
        arrays, 'perimeter', bins=50, min_threshold=10, return_counts=True
    )
    
    print(f"Area exponent: {area_exponent:.3f} ± {area_error:.3f}")
    print(f"Perimeter exponent: {perim_exponent:.3f} ± {perim_error:.3f}")
    
    # Calculate correlation dimension
    print("Calculating correlation dimension...")
    corr_dim, corr_error, corr_lengths, corr_integrals = objscale.ensemble_correlation_dimension(
        arrays, return_C_l=True, point_reduction_factor=200
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
    
    print("\n=== CREATING PLOTS ===")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
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
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== SUMMARY ===")
    print(f"Area exponent: {area_exponent:.3f} ± {area_error:.3f}")
    print(f"Perimeter exponent: {perim_exponent:.3f} ± {perim_error:.3f}")
    print(f"Correlation dimension: {corr_dim:.3f} ± {corr_error:.3f}")
    print(f"Box dimension: {box_dim:.3f} ± {box_error:.3f}")
    print("Individual fractal dimensions:")
    for i, (dim, error) in enumerate(individual_dims):
        print(f"  Array {i+1}: {dim:.3f} ± {error:.3f}")
    
    print("\n=== TEST COMPLETED ===")

if __name__ == "__main__":
    main()