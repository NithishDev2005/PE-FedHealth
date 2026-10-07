import numpy as np


def federated_average(client_updates):

    total_samples = sum(
        update[2] for update in client_updates
    )

    average_coef = None
    average_intercept = None

    for coef, intercept, num_samples in client_updates:

        weight = num_samples / total_samples

        if average_coef is None:

            average_coef = weight * coef
            average_intercept = weight * intercept

        else:

            average_coef += weight * coef
            average_intercept += weight * intercept

    return average_coef, average_intercept