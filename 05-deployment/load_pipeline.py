import urllib.request
import pickle
import sklearn  # ensures sklearn is available when unpickling

# URL and local file name
URL = "https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/refs/heads/master/cohorts/2025/05-deployment/pipeline_v1.bin"
MODEL_FILE = "pipeline_v1.pkl"


def download_pipeline(url: str = URL, filename: str = MODEL_FILE):
    """Download the serialized pipeline if not already saved."""
    print(f"Downloading pipeline from {url} ...")
    urllib.request.urlretrieve(url, filename)
    print(f"Saved as {filename}")
    return filename


def load_pipeline(filename: str = MODEL_FILE):
    """Load the DictVectorizer and model from a pickle file."""
    with open(filename, "rb") as f_in:
        dv, model = pickle.load(f_in)
    print("Pipeline loaded successfully.")
    return dv, model


def predict_single(customer: dict, dv, model) -> float:
    """Generate prediction for a single customer dictionary."""
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


if __name__ == "__main__":
    # Example usage
    filename = download_pipeline()
    dv, model = load_pipeline(filename)

    example = {
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0
    }

    prediction = predict_single(example, dv, model)
    print(f"Prediction: {prediction:.3f}")