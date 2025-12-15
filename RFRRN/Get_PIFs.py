#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
import time
import itertools
import multiprocessing
from multiprocessing import Pool
from datetime import datetime
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from skimage.feature import corner_harris, corner_peaks
import pandas as pd
from tqdm import tqdm
from scipy import linalg
from sklearn.preprocessing import StandardScaler
# Display Chinese fonts
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def extract_tiles(image, tile_size=1024, overlap=256):
    height, width = image.shape[:2]
    tiles = []
    positions = []
    
    # Calculate positions in y direction
    y_positions = list(range(0, height - overlap, tile_size - overlap))
    if y_positions and y_positions[-1] + tile_size < height:
        y_positions.append(height - tile_size)
    
    # Calculate positions in x direction
    x_positions = list(range(0, width - overlap, tile_size - overlap))
    if x_positions and x_positions[-1] + tile_size < width:
        x_positions.append(width - tile_size)
    
    # Extract each tile
    for y in y_positions:
        for x in x_positions:
            if y + tile_size <= height and x + tile_size <= width:
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                positions.append((x, y))
    
    return tiles, positions

def detect_harris_corners(image, min_distance=10, threshold_rel=0.1):

    # Convert to grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Calculate Harris corner response
    harris_response = corner_harris(gray)
    
    # Extract corner peaks
    corners = corner_peaks(harris_response, min_distance=min_distance, 
                          threshold_rel=threshold_rel)
    
    return corners

def phase_correlation(img1, img2):

    # Convert to grayscale
    if len(img1.shape) > 2:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1.copy()
        img2_gray = img2.copy()
    
    # Convert to float32
    img1_float = img1_gray.astype(np.float32)
    img2_float = img2_gray.astype(np.float32)
    
    # Apply Hann window to reduce edge effects
    h, w = img1_gray.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    img1_windowed = img1_float * window
    img2_windowed = img2_float * window
    
    try:
        # Use OpenCV's phase correlation
        # phaseCorrelate returns (shift, response), where shift is (dx, dy)
        shift, response = cv2.phaseCorrelate(img1_windowed, img2_windowed)
        dx, dy = float(shift[0]), float(shift[1])
    except Exception as e:
        print(f"Phase correlation error: {e}")
        # Return zero displacement if error occurs
        dx, dy = 0.0, 0.0
    
    return dy, dx  # Return (dy, dx) for consistency

def extract_patch(image, center_y, center_x, patch_size=256):
    half_size = patch_size // 2
    height, width = image.shape[:2]
    
    # Calculate patch boundaries
    y_start = max(0, center_y - half_size)
    y_end = min(height, center_y + half_size)
    x_start = max(0, center_x - half_size)
    x_end = min(width, center_x + half_size)
    
    # Extract image patch
    patch = image[y_start:y_end, x_start:x_end]
    
    # Pad if patch is smaller than expected (at image edges)
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        if len(image.shape) > 2:
            padded_patch = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
        else:
            padded_patch = np.zeros((patch_size, patch_size), dtype=image.dtype)
        
        # Calculate placement position
        y_offset = half_size - (center_y - y_start)
        x_offset = half_size - (center_x - x_start)
        
        # Place the patch
        padded_patch[y_offset:y_offset+(y_end-y_start), 
                    x_offset:x_offset+(x_end-x_start)] = patch
        return padded_patch
    else:
        # Center crop if patch is exactly or larger than needed
        center_y_local = center_y - y_start
        center_x_local = center_x - x_start
        
        y_start_local = center_y_local - half_size
        y_end_local = center_y_local + half_size
        x_start_local = center_x_local - half_size
        x_end_local = center_x_local + half_size
        
        # Adjust for out-of-bounds
        if y_start_local < 0:
            y_end_local -= y_start_local
            y_start_local = 0
        if x_start_local < 0:
            x_end_local -= x_start_local
            x_start_local = 0
        
        return patch[y_start_local:y_end_local, x_start_local:x_end_local]

def pixel_to_geo(transform, x, y):

    lon, lat = transform * (x, y)
    return lon, lat

def geo_to_pixel(transform, lon, lat):

    x, y = ~transform * (lon, lat)
    return int(x), int(y)

def filter_matches_by_homography(points1, points2, max_distance=3.0):

    if len(points1) < 4:
        # Not enough points to compute homography
        return np.ones(len(points1), dtype=bool), None
    
    # Convert to format required by findHomography
    pts1 = np.float32(points1).reshape(-1, 1, 2)
    pts2 = np.float32(points2).reshape(-1, 1, 2)
    
    # Find homography using RANSAC method
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, max_distance)
    
    return mask.ravel().astype(bool), H

def bilinear_interpolate(image, y, x):

    # Get integer and fractional parts
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Ensure coordinates are within image bounds
    height, width = image.shape[:2]
    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width - 1))
    y0 = max(0, min(y0, height - 1))
    y1 = max(0, min(y1, height - 1))
    
    # Get fractional parts
    dx = x - x0
    dy = y - y0
    
    # Get pixel values at four corners
    if len(image.shape) > 2:
        # Multi-channel image
        val_y0x0 = image[y0, x0, :]
        val_y0x1 = image[y0, x1, :]
        val_y1x0 = image[y1, x0, :]
        val_y1x1 = image[y1, x1, :]
        
        # Interpolate
        val = (1 - dx) * (1 - dy) * val_y0x0 + \
              dx * (1 - dy) * val_y0x1 + \
              (1 - dx) * dy * val_y1x0 + \
              dx * dy * val_y1x1
    else:
        # Single-channel image
        val_y0x0 = image[y0, x0]
        val_y0x1 = image[y0, x1]
        val_y1x0 = image[y1, x0]
        val_y1x1 = image[y1, x1]
        
        # Interpolate
        val = (1 - dx) * (1 - dy) * val_y0x0 + \
              dx * (1 - dy) * val_y0x1 + \
              (1 - dx) * dy * val_y1x0 + \
              dx * dy * val_y1x1
    
    return val

# ---------------- Image Matching Function ----------------tile_size=1024, overlap=256, patch_size=256,

def match_images(img1_path, img2_path, output_dir, output_csv=None, tile_size=256, overlap=64, 
                patch_size=64, min_distance=5, threshold_rel=0.1, homography_threshold=3.0,
                visualize=True):

    # Automatically generate output CSV if not specified
    if output_csv is None:
        img1_name = os.path.splitext(os.path.basename(img1_path))[0]
        img2_name = os.path.splitext(os.path.basename(img2_path))[0]
        output_csv = f"{img1_name}_vs_{img2_name}_matches.csv"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_csv)
    
    # Visualization output path
    if visualize:
        vis_output = output_path.replace(".csv", "_visualization.png")
    
    # Open GeoTIFF images
    with rasterio.open(img1_path) as src1, rasterio.open(img2_path) as src2:
        # Get number of bands
        num_bands1 = src1.count
        num_bands2 = src2.count
        
        # Read images
        img1_bands = src1.read()  # Original bands
        img2_bands = src2.read()  # Original bands
        
        # Convert to (H, W, C) format for OpenCV processing
        img1 = np.transpose(img1_bands, (1, 2, 0))
        img2 = np.transpose(img2_bands, (1, 2, 0))
        
        # Create grayscale images for feature detection
        if img1.shape[2] == 1:
            img1_gray = img1[:, :, 0]
        else:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            
        if img2.shape[2] == 1:
            img2_gray = img2[:, :, 0]
        else:
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Reshape data for processing if single band
        if img1.shape[2] == 1:
            img1_proc = img1[:, :, 0]
        else:
            img1_proc = img1
            
        if img2.shape[2] == 1:
            img2_proc = img2[:, :, 0]
        else:
            img2_proc = img2
        
        # Get coordinate transformation objects
        transform1 = src1.transform
        transform2 = src2.transform
        
        # Extract tiles from the first image
        tiles, positions = extract_tiles(img1_proc, tile_size=tile_size, overlap=overlap)
        
        # Initialize results list
        all_results = []
        
        # Process each tile
        for i, (tile, (tile_x, tile_y)) in enumerate(tqdm(zip(tiles, positions), total=len(tiles), desc="Processing tiles")):
            print(f"\nProcessing tile {i+1}/{len(tiles)}, position: ({tile_x}, {tile_y})")
            
            # Detect Harris corners
            corners = detect_harris_corners(tile, min_distance=min_distance, threshold_rel=threshold_rel)
            
            # Initialize tile results
            tile_results = []
            points1 = []  # Points in image 1
            points2 = []  # Corresponding points in image 2
            
            # Process each corner
            for corner_y, corner_x in corners:
                # Convert corner coordinates to original image coordinate system
                img1_x = tile_x + corner_x
                img1_y = tile_y + corner_y
                
                # Extract patch centered at the corner from the first image
                patch1 = extract_patch(img1_proc, img1_y, img1_x, patch_size=patch_size)
                
                # Convert pixel coordinates to geographic coordinates
                lon, lat = pixel_to_geo(transform1, img1_x, img1_y)
                
                # Convert geographic coordinates to pixel coordinates in the second image
                img2_x, img2_y = geo_to_pixel(transform2, lon, lat)
                
                # Check if point is within the bounds of the second image
                if 0 <= img2_x < img2_proc.shape[1] and 0 <= img2_y < img2_proc.shape[0]:
                    # Extract corresponding patch from the second image
                    patch2 = extract_patch(img2_proc, img2_y, img2_x, patch_size=patch_size)
                    
                    # Perform phase correlation to find offset
                    dy, dx = phase_correlation(patch1, patch2)
                    
                    # Calculate refined matching point in the second image
                    match_x = img2_x + dx
                    match_y = img2_y + dy
                    
                    # Create basic information result dictionary
                    result = {
                        'tile_idx': i,
                        'tile_x': tile_x,
                        'tile_y': tile_y,
                        'img1_x': img1_x,
                        'img1_y': img1_y,
                        'img2_x': match_x,
                        'img2_y': match_y,
                        'lon': lon,
                        'lat': lat
                    }
                    
                    # Use bilinear interpolation to get grayscale values for all bands
                    for b in range(num_bands1):
                        band_data = img1_bands[b]
                        val = bilinear_interpolate(band_data, img1_y, img1_x)
                        result[f'img1_band{b+1}'] = float(val)
                    
                    for b in range(num_bands2):
                        band_data = img2_bands[b]
                        val = bilinear_interpolate(band_data, match_y, match_x)
                        result[f'img2_band{b+1}'] = float(val)
                    
                    tile_results.append(result)
                    points1.append([img1_x, img1_y])
                    points2.append([match_x, match_y])
            
            # Apply homography filtering if enough points
            if len(points1) >= 4:
                points1_array = np.array(points1)
                points2_array = np.array(points2)
                
                # Filter matches using homography constraints
                inlier_mask, H = filter_matches_by_homography(
                    points1_array, points2_array, max_distance=homography_threshold
                )
                
                # Keep only inliers
                filtered_results = [result for result, is_inlier in zip(tile_results, inlier_mask) if is_inlier]
                
                print(f"Tile {i+1}: {len(filtered_results)}/{len(tile_results)} matches remaining after homography filtering")
                
                # Add filtered results to total results
                all_results.extend(filtered_results)
            else:
                # Not enough points for homography filtering, keep all matches
                all_results.extend(tile_results)
        
        # Save results to CSV
        df = pd.DataFrame(all_results)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(all_results)} matches to {output_path}")
        
        # Generate visualization if needed
        if visualize and len(df) > 0:
            visualize_matches(img1_path, img2_path, output_path, output_image=vis_output)
        
        return df, output_path

def visualize_matches(img1_path, img2_path, matches_csv, output_image="matches_visualization.png", 
                     max_points=1000, point_size=15, line_width=1):

    # Read matches
    matches_df = pd.read_csv(matches_csv)
    
    # Limit number of points if needed
    if len(matches_df) > max_points:
        matches_df = matches_df.sample(max_points, random_state=42)
    
    # Open GeoTIFF images
    with rasterio.open(img1_path) as src1, rasterio.open(img2_path) as src2:
        # Read images (use first band for visualization if multi-band)
        img1 = src1.read(1)
        img2 = src2.read(1)
        
        # Normalize images for visualization
        img1_norm = (img1 - img1.min()) / (img1.max() - img1.min())
        img2_norm = (img2 - img2.min()) / (img2.max() - img2.min())
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Display images
        ax1.imshow(img1_norm, cmap='gray')
        ax1.set_title(f'Image 1: {os.path.basename(img1_path)}')
        ax1.axis('off')
        
        ax2.imshow(img2_norm, cmap='gray')
        ax2.set_title(f'Image 2: {os.path.basename(img2_path)}')
        ax2.axis('off')
        
        # Plot matches
        points1_x = matches_df['img1_x'].values
        points1_y = matches_df['img1_y'].values
        points2_x = matches_df['img2_x'].values
        points2_y = matches_df['img2_y'].values
        
        # Scatter plot
        ax1.scatter(points1_x, points1_y, c='red', s=point_size, marker='o')
        ax2.scatter(points2_x, points2_y, c='blue', s=point_size, marker='o')
        
        # Add title with number of matches
        plt.suptitle(f'Matches: {len(matches_df)} points', fontsize=16)
        
        # Save image
        plt.tight_layout()
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_image}")
        
        # Create more detailed visualization showing connecting lines
        fig2, ax = plt.subplots(figsize=(20, 10))
        
        # Calculate dimensions for side-by-side display
        h1, w1 = img1_norm.shape
        h2, w2 = img2_norm.shape
        
        # Create combined image
        combined_width = w1 + w2
        combined_height = max(h1, h2)
        combined_img = np.zeros((combined_height, combined_width))
        
        # Place images side by side
        combined_img[:h1, :w1] = img1_norm
        combined_img[:h2, w1:w1+w2] = img2_norm
        
        # Display combined image
        ax.imshow(combined_img, cmap='gray')
        ax.set_title(f'Matches between images ({len(matches_df)} points)')
        
        # Draw lines connecting matches
        for i in range(len(matches_df)):
            x1, y1 = points1_x[i], points1_y[i]
            x2, y2 = points2_x[i] + w1, points2_y[i]  # Adjust x2 to account for the width of the first image
            ax.plot([x1, x2], [y1, y2], 'c-', linewidth=line_width, alpha=0.5)
        
        # Draw points
        ax.scatter(points1_x, points1_y, c='red', s=point_size, marker='o')
        ax.scatter(points2_x + w1, points2_y, c='blue', s=point_size, marker='o')
        
        # Save detailed visualization
        detailed_output = output_image.replace('.png', '_detailed.png')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(detailed_output, dpi=300, bbox_inches='tight')
        print(f"Detailed visualization saved to {detailed_output}")
        
        # Create heatmap visualization showing point distribution
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Create 2D histogram of point distribution
        h1_bins = min(100, w1//10)  # Adjust bin count based on image size
        h2_bins = min(100, w2//10)
        ax1.hist2d(points1_x, points1_y, bins=[h1_bins, h1_bins], cmap='hot')
        ax1.set_title(f'Point distribution in Image 1')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        ax2.hist2d(points2_x, points2_y, bins=[h2_bins, h2_bins], cmap='hot')
        ax2.set_title(f'Point distribution in Image 2')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
                

        # Save heatmap visualization
        heatmap_output = output_image.replace('.png', '_heatmap.png')
        plt.tight_layout()
        plt.savefig(heatmap_output, dpi=300, bbox_inches='tight')
        print(f"Heatmap visualization saved to {heatmap_output}")
        
        plt.close('all')

# ---------------- IRMAD Filter Functions ----------------

def extract_band_values(df):

    # Find band columns
    band_cols1 = [col for col in df.columns if col.startswith('img1_band')]
    band_cols2 = [col for col in df.columns if col.startswith('img2_band')]
    
    # Check if there are band data
    if not band_cols1 or not band_cols2:
        raise ValueError("No band data found in CSV. Please ensure the input CSV contains band values.")
    
    # Extract band values
    X = df[band_cols1].values
    Y = df[band_cols2].values
    
    return X, Y, band_cols1, band_cols2

def calculate_mad_variates(X, Y, weights=None):

    n_samples = X.shape[0]
    
    # Apply weights (if provided)
    if weights is not None:
        W = np.diag(weights)
        X_w = np.sqrt(weights)[:, np.newaxis] * X
        Y_w = np.sqrt(weights)[:, np.newaxis] * Y
    else:
        W = np.eye(n_samples)
        X_w = X
        Y_w = Y
    
    # Calculate covariance matrices
    Cxx = (X_w.T @ X_w) / n_samples
    Cyy = (Y_w.T @ Y_w) / n_samples
    Cxy = (X_w.T @ Y_w) / n_samples
    
    # Solve generalized eigenvalue problem
    Cxx_inv_sqrt = linalg.inv(linalg.sqrtm(Cxx))
    Cyy_inv_sqrt = linalg.inv(linalg.sqrtm(Cyy))
    
    K = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
    U, s, Vt = linalg.svd(K, full_matrices=False)
    V = Vt.T
    
    # Calculate canonical coefficients
    a = Cxx_inv_sqrt @ U
    b = Cyy_inv_sqrt @ V
    
    # Calculate canonical variables
    U_var = X @ a
    V_var = Y @ b
    
    return a, b, s, U_var, V_var

def calculate_mad_distance(U, V):

    # Calculate squared differences of canonical variables
    Z = (U - V)**2
    
    # Sum over all variables to get chi-square distance
    distances = np.sum(Z, axis=1)
    
    return distances


def calculate_weights(distances, h=1.0):

    # Calculate weights using Gaussian kernel
    weights = np.exp(-distances / (2 * h))
    
    # Normalize weights to sum to n_samples
    weights = weights * (len(weights) / np.sum(weights))
    
    return weights

def filter_saturated_points(df, saturation_threshold=4095, tolerance=50):

    # Find band columns
    band_cols1 = [col for col in df.columns if col.startswith('img1_band')]
    band_cols2 = [col for col in df.columns if col.startswith('img2_band')]
    
    # Create mask for non-saturated points (initially all True)
    mask = np.ones(len(df), dtype=bool)
    
    # Mark point as saturated if any band is close to saturation
    for col in band_cols1 + band_cols2:
        # Check if current band is close to saturation threshold
        saturated = df[col] > (saturation_threshold - tolerance)
        # Use logical AND operation - if any band is saturated, mask becomes False
        mask = mask & (~saturated)
    
    # Apply mask to keep only non-saturated points
    filtered_df = df[mask]
    
    print(f"Removed {len(df) - len(filtered_df)} saturated points from {len(df)} points ({100 * (len(df) - len(filtered_df)) / len(df):.2f}%)")
    
    return filtered_df

def irmad_filter(df, max_iter=20, tol=1e-6, h_factor=0.5, threshold_factor=2.0):

    # Extract band values
    X, Y, band_cols1, band_cols2 = extract_band_values(df)
    
    # Standardize data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_std = scaler_X.fit_transform(X)
    Y_std = scaler_Y.fit_transform(Y)
    
    # Initialize weights
    n_samples = X.shape[0]
    weights = np.ones(n_samples)
    
    # Iterative MAD
    prev_corrs = np.zeros(min(X.shape[1], Y.shape[1]))
    for iteration in range(max_iter):
        # Calculate MAD variables
        a, b, corrs, U, V = calculate_mad_variates(X_std, Y_std, weights)
        
        # Check convergence
        if np.allclose(corrs, prev_corrs, rtol=tol):
            print(f"Converged after {iteration+1} iterations")
            break
        
        # Calculate chi-square distances
        distances = calculate_mad_distance(U, V)
        
        # Adjust kernel bandwidth based on median distance
        h = h_factor * np.median(distances)
        
        # Update weights
        weights = calculate_weights(distances, h)
        
        # Update previous correlations
        prev_corrs = corrs
        
        print(f"Iteration {iteration+1}: Canonical correlations = {corrs}")
    
    # Calculate final chi-square distances
    a, b, corrs, U, V = calculate_mad_variates(X_std, Y_std, weights)
    distances = calculate_mad_distance(U, V)
    
    # Determine threshold based on chi-square distribution
    # For chi-square with k degrees of freedom: mean = k, variance = 2k
    k = len(corrs)
    threshold = k + threshold_factor * np.sqrt(2 * k)
    
    # Filter points
    mask = distances <= threshold
    filtered_df = df.copy()[mask]
    
    print(f"Filtered out {np.sum(~mask)} points from {len(df)} points ({100 * np.sum(~mask) / len(df):.2f}%)")
    
    return filtered_df, distances, threshold

def filter_matches(matches_csv, output_dir, output_csv=None, filter_saturated=True, 
                 saturation_threshold=4095, saturation_tolerance=50, max_iter=20, 
                 tol=1e-6, h_factor=0.5, threshold_factor=2.0, visualize=True):

    # Automatically generate output CSV if not specified
    if output_csv is None:
        base_name = os.path.splitext(os.path.basename(matches_csv))[0]
        output_csv = f"{base_name}_filtered.csv"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_csv)
    
    # Read matches
    print(f"Reading matches from {matches_csv}...")
    df = pd.read_csv(matches_csv)
    
    # Store original number of points
    original_count = len(df)
    
    # Filter saturated points
    if filter_saturated:
        print("Filtering saturated points...")
        df = filter_saturated_points(df, 
                                   saturation_threshold=saturation_threshold,
                                   tolerance=saturation_tolerance)
    
    # Apply IRMAD filtering
    print("Applying IRMAD filtering...")
    filtered_df, distances, threshold = irmad_filter(
        df, 
        max_iter=max_iter,
        tol=tol,
        h_factor=h_factor,
        threshold_factor=threshold_factor
    )
    
    # Save filtered data
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved {len(filtered_df)} filtered matches to {output_path}")
    print(f"Total filtering rate: {100 * (1 - len(filtered_df) / original_count):.2f}%")
    
    # If visualization is needed
    if visualize and len(filtered_df) > 0:
        # Extract image paths from the matches CSV
        base_dir = os.path.dirname(matches_csv)
        img_names = os.path.splitext(os.path.basename(matches_csv))[0].split("_vs_")
        if len(img_names) >= 2:
            # Try to find image files
            img1_pattern = os.path.join(os.path.dirname(matches_csv), f"{img_names[0]}.*")
            img2_pattern = os.path.join(os.path.dirname(matches_csv), f"{img_names[1]}.*")
            
            img1_files = glob.glob(img1_pattern)
            img2_files = glob.glob(img2_pattern)
            
            if img1_files and img2_files:
                img1_path = img1_files[0]
                img2_path = img2_files[0]
                
                # Generate visualization
                vis_output = output_path.replace(".csv", "_visualization.png")
                visualize_filtered_results(df, filtered_df, distances, threshold, 
                                          img1_path, img2_path, 
                                          output_prefix=os.path.splitext(vis_output)[0])
    
    return filtered_df, output_path

def visualize_filtered_results(df, filtered_df, distances, threshold, img1_path, img2_path, output_prefix="irmad_filter"):

    # Plot distance histogram
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.2f}')
    plt.xlabel('Chi-square Distance')
    plt.ylabel('Frequency')
    plt.title('MAD Chi-square Distance Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_histogram.png", dpi=300)
    
    # Plot spatial distribution before and after filtering
    plt.figure(figsize=(12, 10))
    
    # Plot all points
    plt.scatter(df['img1_x'], df['img1_y'], c='lightgray', s=5, label='All points')
    
    # Plot filtered points
    plt.scatter(filtered_df['img1_x'], filtered_df['img1_y'], c='blue', s=5, label='Kept points')
    
    # Plot removed points
    removed_df = df[~df.index.isin(filtered_df.index)]
    plt.scatter(removed_df['img1_x'], removed_df['img1_y'], c='red', s=5, label='Removed points')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Spatial Distribution of Filtered Points')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_spatial.png", dpi=300)
    
    # Plot band values before and after filtering
    band_cols1 = [col for col in df.columns if col.startswith('img1_band')]
    band_cols2 = [col for col in df.columns if col.startswith('img2_band')]
    
    if band_cols1 and band_cols2:
        # Create scatter plots for each band combination
        n_bands1 = len(band_cols1)
        n_bands2 = len(band_cols2)
        
        # Limit to first 3 bands if too many
        if n_bands1 > 3:
            band_cols1 = band_cols1[:3]
            n_bands1 = 3
        if n_bands2 > 3:
            band_cols2 = band_cols2[:3]
            n_bands2 = 3
        
        fig, axes = plt.subplots(n_bands1, n_bands2, figsize=(4*n_bands2, 4*n_bands1))
        
        for i, band1 in enumerate(band_cols1):
            for j, band2 in enumerate(band_cols2):
                ax = axes[i, j] if n_bands1 > 1 and n_bands2 > 1 else axes[j] if n_bands1 == 1 else axes[i]
                
                # Plot kept points
                ax.scatter(filtered_df[band1], filtered_df[band2], c='blue', s=3, alpha=0.5, label='Kept')
                
                # Plot removed points
                ax.scatter(removed_df[band1], removed_df[band2], c='red', s=3, alpha=0.5, label='Removed')
                
                ax.set_xlabel(band1)
                ax.set_ylabel(band2)
                
                if i == 0 and j == 0:
                    ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_bands.png", dpi=300)
    
    plt.close('all')

# ---------------- Multi-process Batch Processing Functions ----------------

def find_geotiff_files(directory):
    # Common GeoTIFF extensions
    extensions = ['tif']
    
    geotiff_files = []
    for ext in extensions:
        geotiff_files.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
    
    return sorted(geotiff_files)

def create_output_directory(base_dir, name="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{name}_{timestamp}")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def process_image_pair(args):

    img1_path, img2_path, output_dir, pair_idx, total_pairs, config = args
    
    # Extract base filename (without extension)
    img1_name = os.path.splitext(os.path.basename(img1_path))[0]
    img2_name = os.path.splitext(os.path.basename(img2_path))[0]
    
    # Create output filenames
    pair_name = f"{img1_name}_vs_{img2_name}"
    matches_csv = os.path.join(output_dir, f"{pair_name}_matches.csv")
    filtered_csv = os.path.join(output_dir, f"{pair_name}_filtered.csv")
    
    print(f"\nProcessing image pair {pair_idx+1}/{total_pairs}: {img1_name} vs {img2_name}")
    
    try:
        # Step 1: Perform image matching
        print("Performing image matching...")
        df, matches_path = match_images(
            img1_path, img2_path, output_dir,
            output_csv=os.path.basename(matches_csv),
            tile_size=config['tile_size'],
            overlap=config['overlap'],
            patch_size=config['patch_size'],
            min_distance=config['min_distance'],
            threshold_rel=config['threshold_rel'],
            homography_threshold=config['homography_threshold'],
            visualize=config['visualize']
        )
        
        # Check if matches were found
        if len(df) == 0:
            print(f"Warning: No matches found for {pair_name}")
            return {
                'img1': img1_name,
                'img2': img2_name,
                'success': False,
                'reason': 'no_matches_found',
                'matches_count': 0,
                'filtered_count': 0
            }
        
        # Step 2: Filter matches
        print("Filtering matches...")
        filtered_df, filtered_path = filter_matches(
            matches_path, output_dir,
            output_csv=os.path.basename(filtered_csv),
            filter_saturated=config['filter_saturated'],
            saturation_threshold=config['saturation_threshold'],
            saturation_tolerance=config['saturation_tolerance'],
            max_iter=config['max_iter'],
            tol=config['tol'],
            h_factor=config['h_factor'],
            threshold_factor=config['threshold_factor'],
            visualize=config['visualize']
        )
        
        # Check if any matches remain after filtering
        if len(filtered_df) == 0:
            print(f"Warning: No matches remaining for {pair_name} after filtering")
            return {
                'img1': img1_name,
                'img2': img2_name,
                'success': False,
                'total_matches': len(df),
                'filtered_matches': 0,
                'removed_percentage': 100
            }
        
        # Return successful results
        return {
            'img1': img1_name,
            'img2': img2_name,
            'success': True,
            'total_matches': len(df),
            'filtered_matches': len(filtered_df),
            'removed_percentage': 100 * (1 - len(filtered_df) / len(df))
        }
    
    except Exception as e:
        print(f"Error processing {pair_name}: {e}")
        return {
            'img1': img1_name,
            'img2': img2_name,
            'success': False,
            'error': str(e)
        }

def create_summary(results, output_dir):

    # Filter out successful results
    successful_results = [r for r in results if r.get('success', False)]
    
    # Create summary DataFrame
    if successful_results:
        summary_df = pd.DataFrame(successful_results)
        summary_path = os.path.join(output_dir, "summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        
        # Create more detailed statistical summary
        print("\nProcessing Summary:")
        print(f"Total image pairs processed: {len(results)}")
        print(f"Successfully processed image pairs: {len(successful_results)}")
        if successful_results:
            print(f"Average matches per pair: {summary_df['total_matches'].mean():.1f}")
            print(f"Average filtered matches per pair: {summary_df['filtered_matches'].mean():.1f}")
            print(f"Average removal percentage: {summary_df['removed_percentage'].mean():.1f}%")
    else:
        print("\nNo data available for summary.")

def batch_process(input_dir, output_dir=None, limit=None, specific_pairs=None, num_processes=None, keep_temp_files=False, **kwargs):

    # Validate input directory
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return False
    
    # Create output directory
    if output_dir is None:
        output_dir = create_output_directory(input_dir)
    print(f"Output directory: {output_dir}")
    
    # Find GeoTIFF files
    geotiff_files = find_geotiff_files(input_dir)
    print(f"Found {len(geotiff_files)} GeoTIFF files in {input_dir}")
    
    if len(geotiff_files) < 2:
        print("Error: Processing requires at least 2 GeoTIFF files.")
        return False
    
    # Determine pairs to process
    if specific_pairs:
        # Read specific pairs from file
        if not os.path.exists(specific_pairs):
            print(f"Error: Specific pairs file '{specific_pairs}' does not exist.")
            return False
        
        with open(specific_pairs, 'r') as f:
            pairs = []
            for line in f:
                if line.strip():
                    img1, img2 = line.strip().split(',')
                    img1_path = os.path.join(input_dir, img1)
                    img2_path = os.path.join(input_dir, img2)
                    
                    # Skip self-pair
                    if img1_path == img2_path:
                        print(f"Warning: Skipping self-pair: {img1}")
                        continue
                        
                    if os.path.exists(img1_path) and os.path.exists(img2_path):
                        pairs.append((img1_path, img2_path))
                    else:
                        print(f"Warning: Cannot find one or both images: {img1}, {img2}")
    else:
        # Generate all possible pairs (itertools.combinations already excludes self-pairs)
        pairs = list(itertools.combinations(geotiff_files, 2))
        
        # Limit number of pairs if needed
        if limit and limit < len(pairs):
            pairs = pairs[:limit]
    
    # Set number of processes
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # Adjust number of processes if fewer pairs than processes
    num_processes = min(num_processes, len(pairs))
    
    print(f"Processing {len(pairs)} image pairs using {num_processes} processes...")
    
    # Prepare multiprocessing arguments
    process_args = []
    for i, (img1_path, img2_path) in enumerate(pairs):
        # Package all parameters as a tuple
        args = (img1_path, img2_path, output_dir, i, len(pairs), kwargs)
        process_args.append(args)
    
    # Start multiprocessing
    results = []
    start_time = time.time()
    
    with Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(process_image_pair, process_args), 
                          total=len(process_args), desc="Processing image pairs"):
            results.append(result)
    
    # Create summary
    create_summary(results, output_dir)
    
    # Print final statistics
    elapsed_time = time.time() - start_time
    successful_count = sum(1 for r in results if r.get('success', False))
    print(f"\nProcessing completed in {elapsed_time:.1f} seconds")
    print(f"Successfully processed {successful_count} pairs out of {len(pairs)}")
    print(f"Results saved to {output_dir}")
    
    # Delete all intermediate match files
    if not keep_temp_files:
        deleted_count = 0
        for pair_idx, (img1_path, img2_path) in enumerate(pairs):
            img1_name = os.path.splitext(os.path.basename(img1_path))[0]
            img2_name = os.path.splitext(os.path.basename(img2_path))[0]
            pair_name = f"{img1_name}_vs_{img2_name}"
            matches_csv = os.path.join(output_dir, f"{pair_name}_matches.csv")
            if os.path.exists(matches_csv):
                try:
                    os.remove(matches_csv)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting temporary file {matches_csv}: {e}")
        print(f"Deleted {deleted_count} temporary match files")
    else:
        print("Keeping temporary CSV files")
    
    return True

# ---------------- Main Program ----------------

def main():
    """Main program entry point"""
    parser = argparse.ArgumentParser(description="GeoTIFF Multi-process Matching and Filtering Tool")
    parser.add_argument("input_dir", help="Directory containing GeoTIFF files")
    parser.add_argument("--output-dir", help="Output directory (default: results_TIMESTAMP in input directory)")
    parser.add_argument("--limit", type=int, help="Limit number of image pairs to process")
    parser.add_argument("--specific-pairs", help="Process specific pairs listed in a text file (one pair per line, format: img1.tif,img2.tif)")
    parser.add_argument("--processes", type=int, help="Number of parallel processes (default: number of CPU cores)")
    
    # Image matching parameters: default tile 1024, overlap 256, matching 256,
    parser.add_argument("--tile-size", type=int, default=256, help="Processing tile size")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between tiles")
    parser.add_argument("--patch-size", type=int, default=64, help="Patch size for matching")
    parser.add_argument("--min-distance", type=int, default=3, help="Minimum distance between Harris corners")
    parser.add_argument("--threshold-rel", type=float, default=0.1, help="Harris corner detection relative threshold")
    parser.add_argument("--homography-threshold", type=float, default=2.0, help="Maximum allowed reprojection error for homography filtering")
    
    # Filtering parameters
    parser.add_argument("--no-filter-saturated", action="store_true", help="Do not filter saturated points")
    parser.add_argument("--saturation-threshold", type=float, default=4095, help="Saturation threshold (typically 4095 for 12-bit data)")
    parser.add_argument("--saturation-tolerance", type=float, default=50, help="Saturation tolerance, higher values mean stricter filtering. For example, 50 means points with value > 4045 will be filtered")
    parser.add_argument("--max-iter", type=int, default=20, help="Maximum number of IRMAD iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="IRMAD convergence tolerance")
    parser.add_argument("--h-factor", type=float, default=0.5, help="IRMAD kernel bandwidth factor")
    parser.add_argument("--threshold-factor", type=float, default=2.0, help="IRMAD chi-square threshold factor")
    
    # Other options
    parser.add_argument("--visualize", action="store_true", help="Generate visualization results")
    
    # Output control
    parser.add_argument("--plot-distribution", action="store_true", help="Generate match point distribution plot")
    parser.add_argument("--plot-heatmap", action="store_true", help="Generate match point heatmap")
    parser.add_argument("--plot-matches", action="store_true", help="Draw matching lines")
    parser.add_argument("--max-plot-points", type=int, default=1000, help="Maximum number of points to use for plotting")
    parser.add_argument("--keep-temp-files", action="store_true", help="Keep temporary CSV files")
    
    args = parser.parse_args()
    
    # Integrate configuration
    config = {
        'tile_size': args.tile_size,
        'overlap': args.overlap,
        'patch_size': args.patch_size,
        'min_distance': args.min_distance,
        'threshold_rel': args.threshold_rel,
        'homography_threshold': args.homography_threshold,
        'filter_saturated': not args.no_filter_saturated,
        'saturation_threshold': args.saturation_threshold,
        'saturation_tolerance': args.saturation_tolerance,
        'max_iter': args.max_iter,
        'tol': args.tol,
        'h_factor': args.h_factor,
        'threshold_factor': args.threshold_factor,
        'visualize': args.visualize
    }
    
    # Run batch processing
    success = batch_process(
        args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        specific_pairs=args.specific_pairs,
        num_processes=args.processes,
        keep_temp_files=args.keep_temp_files,
        **config
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
