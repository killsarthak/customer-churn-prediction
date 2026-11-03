import unittest

import pandas as pd

from app import FEATURE_COLUMNS, calculate_metrics, prepare_training_data, train_model
from src.train_model import build_artifact


def sample_customers(count: int = 100) -> pd.DataFrame:
    rows = []
    for index in range(count):
        churned = index % 2 == 0
        row = {
            "SeniorCitizen": index % 3 == 0,
            "tenure": 4 if churned else 48,
            "MonthlyCharges": 95.0 if churned else 55.0,
            "gender": "Female" if index % 2 else "Male",
            "Partner": "No" if churned else "Yes",
            "Dependents": "No" if churned else "Yes",
            "PhoneService": "Yes",
            "MultipleLines": "Yes" if index % 3 else "No",
            "InternetService": "Fiber optic" if churned else "DSL",
            "OnlineSecurity": "No" if churned else "Yes",
            "OnlineBackup": "No" if churned else "Yes",
            "DeviceProtection": "No" if churned else "Yes",
            "TechSupport": "No" if churned else "Yes",
            "StreamingTV": "Yes" if index % 4 else "No",
            "StreamingMovies": "Yes" if index % 5 else "No",
            "Contract": "Month-to-month" if churned else "Two year",
            "PaperlessBilling": "Yes" if churned else "No",
            "PaymentMethod": "Electronic check" if churned else "Bank transfer",
            "Churn": "Yes" if churned else "No",
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["Churn"])


class ChurnAppTests(unittest.TestCase):
    def test_training_data_normalizes_labels_and_drops_unlabeled_rows(self) -> None:
        frame = sample_customers(20)
        frame.loc[0, "Churn"] = " unknown "

        features, target, labeled_data, dropped_rows = prepare_training_data(frame)

        self.assertEqual(len(features), 19)
        self.assertEqual(len(target), 19)
        self.assertEqual(len(labeled_data), 19)
        self.assertEqual(dropped_rows, 1)
        self.assertEqual(set(target.unique()), {0, 1})

    def test_pipeline_trains_and_scores_held_out_customers(self) -> None:
        bundle = train_model(sample_customers())
        probabilities = bundle["model"].predict_proba(bundle["x_test"])[:, 1]
        metrics = calculate_metrics(bundle["y_test"], probabilities, 0.40)

        self.assertEqual(len(probabilities), len(bundle["y_test"]))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        self.assertGreaterEqual(metrics["recall"], 0)
        self.assertLessEqual(metrics["recall"], 1)
        self.assertGreaterEqual(metrics["roc_auc"], 0)
        self.assertLessEqual(metrics["roc_auc"], 1)

    def test_missing_feature_is_reported(self) -> None:
        frame = sample_customers().drop(columns=["Contract"])

        with self.assertRaisesRegex(ValueError, "Contract"):
            prepare_training_data(frame)

    def test_saved_artifact_scores_customers_without_training_rows(self) -> None:
        artifact = build_artifact(sample_customers())
        probabilities = artifact["model"].predict_proba(
            sample_customers(2)[FEATURE_COLUMNS]
        )

        self.assertEqual(probabilities.shape, (2, 2))
        self.assertEqual(artifact["model_type"], "LogisticRegression")
        self.assertIn("0.40", artifact["threshold_metrics"])
        self.assertNotIn("training_features", artifact)


if __name__ == "__main__":
    unittest.main()
