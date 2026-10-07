import numpy as np


def secure_average(
    client_updates
):

    if not client_updates:

        raise ValueError(
            "No client updates were provided."
        )


    total_samples = sum(
        update[2]
        for update in client_updates
    )


    if total_samples <= 0:

        raise ValueError(
            "Total number of samples must be positive."
        )


    weighted_coef = None

    weighted_intercept = None


    for (
        coef,
        intercept,
        samples
    ) in client_updates:

        weight = (
            samples
            / total_samples
        )


        if weighted_coef is None:

            weighted_coef = (
                coef * weight
            )

            weighted_intercept = (
                intercept * weight
            )

        else:

            weighted_coef += (
                coef * weight
            )

            weighted_intercept += (
                intercept * weight
            )


    return (
        weighted_coef,
        weighted_intercept
    )