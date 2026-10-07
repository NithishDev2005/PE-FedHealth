import numpy as np


def privatize_parameters(
    parameters,
    max_norm=1.0,
    noise_scale=0.05
):

    coef, intercept = parameters


    # =====================================================
    # COMBINE PARAMETERS
    # =====================================================

    combined = np.concatenate([
        coef.flatten(),
        intercept.flatten()
    ])


    # =====================================================
    # CLIP NORM
    # =====================================================

    norm = np.linalg.norm(
        combined
    )


    if norm > max_norm:

        combined = (
            combined
            * max_norm
            / norm
        )


    # =====================================================
    # ADD GAUSSIAN NOISE
    # =====================================================

    noise = np.random.normal(
        loc=0.0,
        scale=noise_scale,
        size=combined.shape
    )


    protected = (
        combined + noise
    )


    # =====================================================
    # RESTORE SHAPES
    # =====================================================

    coef_size = coef.size

    protected_coef = (
        protected[:coef_size]
        .reshape(coef.shape)
    )

    protected_intercept = (
        protected[coef_size:]
        .reshape(intercept.shape)
    )


    return (
        protected_coef,
        protected_intercept
    )