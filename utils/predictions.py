
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Locate the saved model bundle
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "water_quality_model.pkl"

# Load model and threshold
bundle = joblib.load(MODEL_PATH)

best_model = bundle["model"]
BEST_THRESHOLD = bundle["threshold"]

# Feature names and order used during training
FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]


def analyze_water(input_data):
    """
    input_data: dictionary containing water-quality measurements.
    Returns prediction, probability and uncertainty information.
    """

    sample_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    # Get probability for potable class (class 1)
    probabilities = best_model.predict_proba(sample_df)[0]
    classes = list(best_model.classes_)
    potable_index = classes.index(1)

    potable_probability = float(probabilities[potable_index])

    # Apply optimized threshold consistently
    prediction = int(potable_probability >= BEST_THRESHOLD)

    # Calculate normalized binary entropy
    p = np.clip(probabilities, 1e-10, 1.0)
    entropy = -np.sum(p * np.log(p)) / np.log(len(p))

    if entropy >= 0.80:
        uncertainty = "High"
    elif entropy >= 0.50:
        uncertainty = "Moderate"
    else:
        uncertainty = "Low"

    return {
        "prediction": prediction,
        "potable_probability": potable_probability,
        "non_potable_probability": float(
            probabilities[classes.index(0)]
        ),
        "threshold": BEST_THRESHOLD,
        "uncertainty_score": float(entropy),
        "uncertainty_level": uncertainty
    }
