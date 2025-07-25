# RF Imaging Reconstruction Project

This project implements advanced RF imaging reconstruction techniques, combining classical and neural network-based approaches. The system simulates RF measurements from 2D targets and reconstructs them using optimization and deep learning methods.

## Problem Setup
- **Target:** 2D reflectivity distribution on the z=z₀ plane (parallel to xy plane)
- **Terminals:** Each terminal has a Uniform Rectangular Array (URA) transmitter and receiver, with configurable spacing, rotation, and offsets.
- **Operation Modes:**
  - *Monostatic:* Each Rx receives from its own Tx
  - *Multistatic:* Each Rx can receive from all Tx terminals
- **Measurement Model:** y = Ax, where x is the vectorized target reflectivity

## Key Concepts
- **Resolution Distinction:**
  - *Measurement Grid:* High resolution (e.g., 128×128) for accurate forward modeling
  - *Reconstruction Grid:* Coarser resolution (e.g., 32×32 or 64×64) for computational efficiency
  - *Target Bounds:* 1m × 1m physical area (configurable)

## Reconstruction Approaches
- **Classical Methods:**
  - Back Projection (BP)
  - LASSO with wavelet regularization
- **VAE-Based Methods:**
  - Method 1: x̂ = argmin_x ‖y - Ax‖₂² + μ‖x - ψ(x)‖₂²
  - Method 2: x̂ = G_θ(argmin_z ‖y - AG_θ(z)‖₂² + λR(G_θ(z)))
- **Advanced Solvers:**
  - OAMP (Orthogonal Approximate Message Passing)
  - Gradient-based methods (Adam, SGD)

## Data
- **Input Images:** Black and white 2D images (e.g., from `data_x/`), used as targets for simulation and reconstruction.
- **Resolutions:**
  - Measurement: High (e.g., 128x128)
  - Reconstruction/Training: Coarse (e.g., 64x64)
