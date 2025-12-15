RF-RRN: Radiometric Calibration for Remote Sensing Imagery This repository implements RF-RRN (Reference-Free Radiometric Normalization), a robust framework for radiometric calibration of multi-temporal remote sensing images—without requiring a fixed reference image. The method leverages pseudo-invariant features (PIFs) to achieve consistent radiometric alignment across time, making it especially suitable for dynamic or unstable regions.

📌 Key Features Reference-free: No need to select a stable reference image. Robust PIF selection: Automatic extraction and outlier filtering of pseudo-invariant features. Flexible input: Supports multi-spectral or RGB nighttime light imagery. Two-stage pipeline: Modular design for easy integration and debugging. 
🚀 Workflow Overview The calibration process consists of two main steps:

get_PIFs.py – Extract and validate pseudo-invariant features Input: A set of co-registered remote sensing images Output: A cleaned set of reliable PIFs calculate_RFRRN.py – Compute radiometric normalization parameters Input: Validated PIFs (organized as a dictionary) Output: Calibration coefficients for each image pair 

⚠️ Important Precondition

Ensure that all input images are rigorously geo-registered to avoid spatial misalignment. Misaligned pixels will severely degrade PIF detection and calibration accuracy.

Step-by-step Procedure: Run get_PIFs.py to extract matched pseudo-invariant features (PIFs) from your image stack. The script automatically performs outlier detection to filter unreliable candidates. The validated PIFs are saved in a structured dictionary format. Pass this dictionary to calculate_RFRRN.py to compute the final radiometric normalization parameters.