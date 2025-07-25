# src/dashboard/assets/help_texts.py

# =============================================================================
# GENERAL GUIDELINES
# =============================================================================

RESOLUTION_GUIDELINE = r"""
### How to Choose Your Grid Size

The quality of the reconstructed image depends heavily on the **Reconstruction Grid** size. If the grid is too coarse, you will lose details and suffer from an artifact called **aliasing**. If it's too fine, the computation will be unnecessarily slow.

The key is to match the grid's pixel size to the system's theoretical **cross-range resolution**.

#### Cross-Range Resolution Theory
The cross-range resolution is the smallest distance between two objects that the system can distinguish. The formula is:
$$ \Delta c \approx \frac{\lambda R}{L} $$

- **$ \lambda $ (Wavelength)**: The wavelength of the carrier signal. Shorter wavelengths (higher frequencies) lead to better resolution.
- **$ R $ (Range)**: The distance from the terminal to the target. Resolution degrades at longer distances.
- **$ L $ (Aperture)**: The physical size of the receiver array. A larger array provides better resolution.

To avoid aliasing, your grid's pixel size should be, at most, half of the resolution (Nyquist criterion). This gives us a recommended minimum grid size:
$$ \text{rec\_grid} \ge \frac{2 \cdot L \cdot \text{SceneWidth}}{\lambda R} $$

Below is a live calculation of this recommendation based on your current terminal and target settings.
"""

MEASUREMENT_MODEL_GUIDELINE = r"""
### Understanding the Measurement Model

The entire imaging process is described by a fundamental linear equation that connects the target scene to the received signals:
$$ \mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{n} $$

Here is a breakdown of each component:

---
#### **$ \mathbf{x} $ : The Reflectivity Map**
- This is a long vector representing the 2D target scene we want to image.
- It is created by flattening the **Reconstruction Grid** into a 1D vector. If the grid is `64x64`, then `x` is a vector with `4096` elements.
- Each element of `x` corresponds to the reflectivity (brightness) of a single pixel in the scene.
- Our goal in reconstruction is to find a good estimate of `x`.

---
#### **$ \mathbf{A} $ : The Measurement Matrix**
- This is the heart of the physics model. It describes how every point in the scene contributes to every measurement at the receiver.
- Each **column** of `A` represents the complete channel response from a **single pixel** on the grid to **all `N_rx` receiver antennas** for a single transmission.
- The response `h` from a transmitter at position $p_{tx}$ to a receiver at $p_{rx}$ via a scatterer at $p_{s}$ is calculated based on path length and attenuation. The complex-valued response is given by:

$$ h(p_{tx}, p_{rx}, p_s) = \frac{e^{-j \cdot k \cdot (d_{tx} + d_{rx})}}{4\pi \cdot d_{tx} \cdot d_{rx}} $$

- $d_{tx} = \|p_{tx} - p_s\|$ is the transmitter-scatterer distance.
- $d_{rx} = \|p_{rx} - p_s\|$ is the receiver-scatterer distance.
- $k = 2\pi / \lambda$ is the wavenumber.
- Each entry in `A` is also scaled by the grid cell area (`dx * dy`) to approximate a continuous surface integral.

---
#### **$ \mathbf{y} $ : The Measurement Vector**
- This is a vector containing all the complex-valued signals measured at the receiver array.
- Its total length is `N_tx * N_rx`, where `N_tx` is the number of transmitter elements and `N_rx` is the number of receiver elements.
- It is formed by stacking the measurements for each transmitter one after another.

---
#### **Channel Effects**

**$ \mathbf{n} $ : Additive Noise**
- This vector represents random thermal noise in the receiver hardware.
- When enabled, it is modeled as **Additive White Gaussian Noise (AWGN)**.
- Its power is determined by the Signal-to-Noise Ratio (SNR) you set. A lower SNR means more powerful noise, making reconstruction harder.

**Fading**
- When enabled, this simulates fluctuations in signal strength due to complex, multi-path propagation not captured by the simple line-of-sight model in `A`.
- It is modeled as **Rayleigh Fading**, where the clean signal `Ax` is multiplied by a complex random variable `h`:
$$ \mathbf{y} = h \cdot (\mathbf{A}\mathbf{x}) + \mathbf{n} $$
"""

# =============================================================================
# RECONSTRUCTION METHOD OVERVIEWS
# =============================================================================

BP_OVERVIEW = r"""
#### Back Projection (BP)
This method is a fundamental and straightforward imaging technique based on the principle of a matched filter. It essentially "smears" the received measurements back across the imaging grid.

**Mathematical Formulation:**
The reconstruction `x_hat` is obtained by applying the Hermitian transpose (conjugate transpose) of the measurement matrix **A** to the measurement vector **y**:
$$ \hat{\mathbf{x}} = \mathbf{A}^H \mathbf{y} $$

- **Pros:** Computationally very fast and simple to implement. It does not require iterative solvers or parameter tuning.
- **Cons:** It does not actually solve the inverse problem `y=Ax`. The result is often blurry and suffers from significant artifacts, especially in the presence of noise or limited measurements. It provides a baseline image but lacks the clarity of regularized methods.
"""

L2_OVERVIEW = r"""
#### L2 Regularization (Ridge Regression)
This is a classical regularization method used to solve ill-posed inverse problems. It adds a penalty term based on the squared L2-norm (Euclidean length) of the solution vector, which encourages solutions with small-magnitude values.

**Mathematical Formulation:**
Ridge Regression finds the reflectivity map `x` that minimizes the following objective function:
$$ \hat{\mathbf{x}} = \arg \min_{\mathbf{x}} \underbrace{\| \mathbf{A}\mathbf{x} - \mathbf{y} \|_2^2}_{\text{Data Fidelity}} + \underbrace{\alpha \|\mathbf{x}\|_2^2}_{\text{Regularization}} $$

- **Data Fidelity Term:** Ensures the reconstructed image `x` is consistent with the actual measurements `y`.
- **Regularization Term:** Penalizes solutions with large reflectivity values. This helps to stabilize the inversion process, prevent noise amplification, and produce smoother, more physically plausible images.
- **α (alpha):** The regularization parameter controls the trade-off between these two terms. A larger `α` leads to a smoother but potentially blurrier reconstruction.
"""

L1_OVERVIEW = r"""
#### L1 Regularization (LASSO)
LASSO (Least Absolute Shrinkage and Selection Operator) is a powerful regularization technique that promotes **sparsity** in the solution. This means it encourages the reconstructed image (or its representation) to have many zero-valued entries. This is extremely useful for scenarios where the target is known to be sparse, such as imaging a few, small point-like scatterers.

**Mathematical Formulation (Synthesis Form):**
The problem is formulated to solve for a set of sparse coefficients `s`, where the image `x` is synthesized from them via a dictionary `D` (e.g., an inverse wavelet transform).
$$ \hat{\mathbf{s}} = \arg \min_{\mathbf{s}} \underbrace{\| \mathbf{A}(\mathbf{Ds}) - \mathbf{y} \|_2^2}_{\text{Data Fidelity}} + \underbrace{\alpha \|\mathbf{s}\|_1}_{\text{Sparsity Regularization}} $$
The final image is then `x_hat = D*s_hat`.

- The **Data Fidelity** term ensures the image reconstructed from the sparse coefficients matches the measurements.
- The **Sparsity Regularization** term, the L1-norm `||s||_1`, forces most of the coefficients in `s` to become exactly zero.
"""

EN_OVERVIEW = r"""
#### Elastic Net Regularization
Elastic Net is a powerful regularization technique that linearly combines the L1 (LASSO) and L2 (Ridge) penalties. This allows it to inherit the benefits of both methods, making it versatile for complex or correlated data.

**Mathematical Formulation:**
Elastic Net finds the solution that minimizes:
$$ \hat{\mathbf{x}} = \arg \min_{\mathbf{x}} \| \mathbf{A}\mathbf{x} - \mathbf{y} \|_2^2 + \alpha \rho \|\mathbf{Dx}\|_1 + \frac{\alpha(1-\rho)}{2} \|\mathbf{x}\|_2^2 $$
- `ρ` corresponds to the `l1_ratio` parameter, controlling the mix between L1 and L2 penalties.
- `D` is the sparsifying transform (Identity or Wavelet).
- It can produce a sparse solution while maintaining the smoothness and grouping effect of Ridge regression, which is useful for handling correlated features.
"""

TV_OVERVIEW = r"""
#### Total Variation (TV) Regularization
Total Variation regularization is designed to reconstruct images that are **piecewise-constant**—composed of flat regions separated by sharp edges. It works by penalizing the gradient of the image, which encourages large areas to be flat (zero gradient) while allowing sharp jumps at edges.

**Mathematical Formulation:**
TV regularization minimizes the following objective function:
$$ \hat{\mathbf{x}} = \arg \min_{\mathbf{x}} \underbrace{\| \mathbf{A}\mathbf{x} - \mathbf{y} \|_2^2}_{\text{Data Fidelity}} + \underbrace{\alpha \cdot \mathrm{TV}(\mathbf{x})}_{\text{Regularization}} $$
The **Total Variation** term, `TV(x)`, is the sum of the magnitudes of the image gradient at every pixel.
"""

NN_OVERVIEW = r"""
#### Learned Regularizer (Neural Network Prior)
Learned Regularizers use a neural network, trained on a dataset of representative images, to impose a powerful, learned prior on the reconstruction. This approach can capture complex target structures beyond the simple assumptions of classical methods. Three main formulations are supported:

1.  **Implicit Prior (Generator-based)**
    Solve for a latent code `z` that minimizes the objective. The prior is implicitly enforced by constraining the solution to the output manifold of the generator `G_theta`.
    $$ \hat{\mathbf{x}} = G_\theta\left( \arg\min_{\mathbf{z}} \frac{1}{2\sigma^2} \| \mathbf{y} - A G_\theta(\mathbf{z}) \|_2^2 + \lambda \|\mathbf{z}\|_2^2 \right) $$

2.  **Explicit Prior on Output**
    Similar to the implicit prior, but adds an explicit regularization term `R(.)` on the generated output image itself.
    $$ \hat{\mathbf{x}} = G_\theta\left( \arg\min_{\mathbf{z}} \frac{1}{2\sigma^2} \| \mathbf{y} - A G_\theta(\mathbf{z}) \|_2^2 + \mathcal{R}(G_\theta(\mathbf{z})) \right) $$

3.  **Autoencoder Penalty (Projection-based)**
    Optimize in the image space `x`, but add a penalty that encourages the solution to lie on or near the manifold learned by a full autoencoder `ψ(x) = D(E(x))`.
    $$ \hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \| \mathbf{y} - A \mathbf{x} \|_2^2 + \mu \| \mathbf{x} - \psi(\mathbf{x}) \|_2^2 $$
"""

# =============================================================================
# PARAMETER-SPECIFIC HELP TEXTS
# =============================================================================

L1_BASIS_HELP = {
    'Identity': r"""
**`Identity` Basis (Pixel Sparsity):**
This is the most straightforward application of LASSO, where the L1 penalty is applied directly to the image's pixel values.
- **Assumption:** The image itself is sparse, meaning it consists of a few bright pixels on a mostly black background.
- **Best for:** Scenarios with a small number of point targets or tiny, distinct objects.
""",
    'Wavelet': r"""
**`Wavelet` Basis (Wavelet Sparsity):**
This is a much more powerful approach that assumes the image's **Wavelet Transform** is sparse, not the image itself.
- **Assumption:** Most natural images are *compressible*, meaning they can be accurately represented by a small number of significant wavelet coefficients. The L1 penalty is applied to these coefficients.
- **Best for:** Reconstructing complex, non-sparse targets like geometric shapes. It can reconstruct a full, detailed image by finding a sparse representation in the wavelet domain. This is the core idea behind compressed sensing.
"""
}

EN_L1_RATIO_HELP = r"""
**L1 Ratio (`l1_ratio`)**
This parameter, often denoted as `ρ` (rho), controls the **mix** between the L1 and L2 penalties, and must be between 0 and 1.
- **`l1_ratio` = 1 (Pure LASSO):** The penalty is purely L1, enforcing maximum sparsity.
- **`l1_ratio` = 0 (Pure Ridge):** The penalty is purely L2, resulting in a smooth, non-sparse solution.
- **0 < `l1_ratio` < 1 (Elastic Net):** The model is a blend of both. Values closer to 1 (e.g., 0.9) are very sparse but more stable than pure LASSO, while values closer to 0 (e.g., 0.1) are much smoother.
"""

TV_NORM_HELP = r"""
**TV Norm**
This parameter defines how the magnitude of the gradient is calculated at each pixel `(i,j)`.
- **`isotropic` (L2-norm):** The gradient magnitude is the Euclidean distance:
$$ \|\nabla\mathbf{x}_{i,j}\| = \sqrt{ ( \nabla_h )^2 + ( \nabla_v )^2 } $$
This method is rotationally invariant, penalizing gradients equally in all directions and leading to smooth, rounded corners. It is generally the preferred choice.

- **`anisotropic` (L1-norm):** The gradient magnitude is the Manhattan distance:
$$ \|\nabla\mathbf{x}_{i,j}\| = | \nabla_h | + | \nabla_v | $$
This method penalizes horizontal and vertical gradients independently, which can preserve axis-aligned edges but may create "staircase" artifacts on diagonal edges.
"""

WAVELET_LEVEL_HELP = r"""
The **Decomposition Level** determines how many times the wavelet transform is recursively applied.
- **Level 1**: The image is split into one low-frequency *approximation* band and three high-frequency *detail* bands.
- **Higher Levels**: The transform is applied again to the low-frequency band from the previous level.
A higher level captures features at coarser scales, which can lead to a sparser representation but increases computational cost. A level between 2 and 4 is often a good starting point.
"""

VAE_TRAINING_EXPLANATION = r"""
### Variational Autoencoder (VAE): Concepts and Training

A **Variational Autoencoder (VAE)** is a deep generative model that learns to compress data (like images) into a continuous, structured *latent space* and then reconstruct it. By training on a dataset of valid targets, a VAE learns the statistical properties of realistic images, providing a powerful prior for reconstruction tasks.

#### **VAE Structure**
- **Encoder**: Maps an input image $\mathbf{x}$ to a probability distribution in the latent space, defined by a mean $\boldsymbol{\mu}_\phi(\mathbf{x})$ and a variance $\boldsymbol{\sigma}_\phi^2(\mathbf{x})$.
- **Decoder**: Maps a sample $\mathbf{z}$ from the latent space back to the image space, generating a new image $\hat{\mathbf{x}}$.

#### **The VAE Objective (ELBO)**
VAE training maximizes the **Evidence Lower Bound (ELBO)**, which consists of two terms:
1.  **Reconstruction Loss:** Measures how well the VAE reconstructs an input image (e.g., using Binary Cross-Entropy or Mean Squared Error).
2.  **KL Divergence:** A regularization term that forces the latent space distribution to be close to a prior (usually a standard normal distribution), which ensures the latent space is smooth and continuous.

The combined loss can be weighted with a parameter $\beta$:
$$ \mathcal{L} = \text{Reconstruction Loss} + \beta \cdot \mathrm{KL} $$
- $\beta=1$ is a standard VAE. $\beta>1$ encourages a more disentangled latent space, while $\beta<1$ prioritizes reconstruction fidelity.
"""

# =============================================================================
# DETAILED EXPLANATIONS: METHODS AND PARAMETERS
# =============================================================================

BP_DETAILS = r"""
**Back Projection (BP) — In-Depth**

Back Projection reconstructs the image by simply applying the Hermitian (conjugate transpose) of the measurement matrix to the measurement vector:
$$
\hat{\mathbf{x}} = \mathbf{A}^H \mathbf{y}
$$

- **No explicit regularization:** This is not a true inversion. The result is a "smeared" image, which is often blurry.
- **No user parameters:** There is nothing to tune.

**When to use:** As a baseline for comparison, or for fast, qualitative previews.

**Limitations:**  
- Artifacts and strong blurring, especially with noisy or limited data.
"""

L2_DETAILS = r"""
**L2 Regularization (Ridge) — In-Depth**

**Objective:** Minimize:
$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{y}\|_2^2 + \alpha \|\mathbf{x}\|_2^2
$$

- **Data Fidelity:** $\|\mathbf{A}\mathbf{x} - \mathbf{y}\|_2^2$ ensures the solution matches the data.
- **L2 Regularization:** $\alpha \|\mathbf{x}\|_2^2$ penalizes large pixel values, controlling overfitting/noise.

**Key Parameters:**
- **Regularization Strength ($\alpha$):** Controls smoothness. High $\alpha$ = smoother, but possibly over-blurred.
- **Solver:** Numerical backend. Usually "auto" suffices, but others may be faster for large matrices.
    - _svd_: Singular Value Decomposition (stable for small/medium problems).
    - _lsqr_: Iterative; better for large problems.

**Tips:**
- If you see too much smoothing, try lowering $\alpha$.
- If the reconstruction is noisy or unstable, increase $\alpha$.
"""

L1_DETAILS = r"""
**L1 Regularization (LASSO) — In-Depth**

**Objective:**
$$
\hat{\mathbf{s}} = \arg\min_{\mathbf{s}} \|\mathbf{A}(\mathbf{D}\mathbf{s}) - \mathbf{y}\|_2^2 + \alpha \|\mathbf{s}\|_1
$$
where $\mathbf{x} = \mathbf{D}\mathbf{s}$ and $\mathbf{D}$ is either the identity or a wavelet transform.

- **L1 Regularization:** Promotes sparsity in $\mathbf{s}$. Most coefficients go to zero.
- **Sparsifying Basis ($\mathbf{D}$):** "Identity" means pixels; "Wavelet" means the L1 is applied in the wavelet domain.
- **Wavelet Name/Level:** Select which wavelet family and how many recursive decompositions are used (higher = sparser, more compressed).
- **Max Iterations:** Number of ISTA steps (higher = more accurate, but slower).
- **Step Size:** The ISTA gradient step (usually $1/L$, where $L$ is the Lipschitz constant of $\mathbf{A}^T\mathbf{A}$).

**Parameter Effects:**
- **$\alpha$ (Sparsity):** Higher = more zeros, possible loss of detail. Lower = risk of noise/artifacts.
- **Wavelet:** Use for non-sparse or geometric shapes; "Identity" for point scatterers.
- **Step Size:** Too large = divergence; too small = slow convergence.
"""

EN_DETAILS = r"""
**Elastic Net — In-Depth**

**Objective:**
$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{y}\|_2^2 + \alpha \rho \|\mathbf{D}\mathbf{x}\|_1 + \frac{\alpha(1-\rho)}{2} \|\mathbf{x}\|_2^2
$$
where $0 \leq \rho \leq 1$.

- **L1 Ratio ($\rho$):** Controls mix between sparsity (L1) and smoothness (L2).
    - $\rho = 1.0$: Pure LASSO.
    - $\rho = 0.0$: Pure Ridge.
    - $0 < \rho < 1$: Hybrid.
- **Other parameters:** Same as LASSO and Ridge.
- **Max Iterations, Step Size:** Control the optimizer as above.

**Use When:**
- Your image is sparse *and* you want some smoothness, or when LASSO gives unstable solutions.
"""

TV_DETAILS = r"""
**Total Variation (TV) — In-Depth**

**Objective:**
$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{y}\|_2^2 + \alpha\, \mathrm{TV}(\mathbf{x})
$$

**Total Variation:** 
$$
\mathrm{TV}(\mathbf{x}) = \sum_{i,j} \| \nabla \mathbf{x}_{i,j} \|
$$

- **Norm Type:**  
    - _isotropic (L2)_: $ \|\nabla \mathbf{x}_{i,j}\| = \sqrt{ (\nabla_h)^2 + (\nabla_v)^2 } $ (preferred for natural images)
    - _anisotropic (L1)_: $ \|\nabla \mathbf{x}_{i,j}\| = |\nabla_h| + |\nabla_v| $
- **Max Iterations:** Controls how long the optimizer runs. Higher = better solution, but slower.
- **Solver:** Backend algorithm (e.g., SCS, ECOS, CVXOPT). SCS is robust but slow.

**Parameter Effects:**
- **$\alpha$ (TV Strength):** Higher = blockier images, more piecewise-constant. Too high = loss of detail.
- **Norm:** Isotropic is smoother/rounder; anisotropic preserves axis-aligned features.
"""

NN_DETAILS = r"""
**Learned Regularizer (Neural Network Prior) — In-Depth**

**What it does:** Uses a neural network (typically a VAE) trained on target images to constrain reconstruction to plausible images.

**Formulations:**
1. **Implicit Prior (Generator-based)**
   - Minimize
     $$
     \hat{\mathbf{x}} = G_\theta\left( \arg\min_{\mathbf{z}} \frac{1}{2\sigma^2} \| \mathbf{y} - A G_\theta(\mathbf{z}) \|_2^2 + \lambda \|\mathbf{z}\|_2^2 \right)
     $$
   - **Latent Norm Penalty ($\lambda$):** Penalizes large latent codes, regularizing the optimization.

2. **Explicit Prior on Output**
   - Minimize
     $$
     \hat{\mathbf{x}} = G_\theta\left( \arg\min_{\mathbf{z}} \frac{1}{2\sigma^2} \| \mathbf{y} - A G_\theta(\mathbf{z}) \|_2^2 + \alpha \mathcal{R}(G_\theta(\mathbf{z})) \right)
     $$
   - **Output Regularizer Weight ($\alpha$):** Controls strength of explicit penalty (often L2 norm) on generated image.

3. **Autoencoder Penalty (Projection-based)**
   - Minimize
     $$
     \hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \| \mathbf{y} - A \mathbf{x} \|_2^2 + \mu \| \mathbf{x} - \psi(\mathbf{x}) \|_2^2
     $$
   - **AE Penalty Weight ($\mu$):** Controls how strictly reconstructions are forced to the learned data manifold.

**Other Key Parameters:**
- **Max Iterations:** How long the optimizer runs (higher = better, but slower).
- **Step Size:** Learning rate for optimizer (GD/AMP).
- **Solver:** "Gradient Descent" (vanilla Adam/SGD) or "AMP" (Approximate Message Passing-style).

**Tips:**
- If your reconstructions are blurry, try lowering penalty weights.
- If they are unstable or unphysical, increase penalty weights or max iterations.
"""




