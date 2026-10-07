import numpy as np

from sklearn.linear_model import SGDClassifier

from privacy.differential_privacy import (
    privatize_parameters
)


class FederatedClient:

    def __init__(
        self,
        X,
        y,
        client_id
    ):

        self.X = X

        self.y = y

        self.client_id = client_id

        self.model = SGDClassifier(
            loss="log_loss",
            random_state=42,
            max_iter=1000,
            tol=1e-3
        )


    # =====================================================
    # LOCAL TRAINING
    # =====================================================

    def train(
        self,
        global_parameters,
        classes
    ):

        # -----------------------------------------------
        # Load global model
        # -----------------------------------------------

        if global_parameters is not None:

            coef, intercept = (
                global_parameters
            )

            self.model.coef_ = (
                coef.copy()
            )

            self.model.intercept_ = (
                intercept.copy()
            )

            self.model.classes_ = (
                classes.copy()
            )


        # -----------------------------------------------
        # Balanced sample weights
        # -----------------------------------------------

        sample_weights = np.ones(
            len(self.y),
            dtype=float
        )


        unique_classes, class_counts = (
            np.unique(
                self.y,
                return_counts=True
            )
        )


        number_of_classes = len(
            unique_classes
        )

        number_of_samples = len(
            self.y
        )


        for (
            class_value,
            class_count
        ) in zip(
            unique_classes,
            class_counts
        ):

            if class_count > 0:

                weight = (
                    number_of_samples
                    /
                    (
                        number_of_classes
                        *
                        class_count
                    )
                )

                sample_weights[
                    self.y == class_value
                ] = weight


        # -----------------------------------------------
        # Local model training
        # -----------------------------------------------

        self.model.partial_fit(
            self.X,
            self.y,
            classes=classes,
            sample_weight=sample_weights
        )


        # -----------------------------------------------
        # Get parameters
        # -----------------------------------------------

        coef = self.model.coef_.copy()

        intercept = (
            self.model.intercept_.copy()
        )


        local_parameters = (
            coef,
            intercept
        )


        # -----------------------------------------------
        # Privacy protection
        # -----------------------------------------------

        protected_parameters = (
            privatize_parameters(
                local_parameters,
                max_norm=1.0,
                noise_scale=0.05
            )
        )


        protected_coef = (
            protected_parameters[0]
        )

        protected_intercept = (
            protected_parameters[1]
        )


        return (
            protected_coef,
            protected_intercept,
            len(self.X)
        )