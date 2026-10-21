import numpy as np
import pandas as pd
import pytest

from src.features import MODEL_FEATURE_COLUMNS, engineer_features, prepare_inference


def sample() -> pd.DataFrame:
    return pd.DataFrame({
        "modelMag_u": [20.0], "modelMag_g": [19.0], "modelMag_r": [18.5],
        "modelMag_i": [18.0], "modelMag_z": [17.5],
    })


def test_engineer_features_calculates_colors():
    features = engineer_features(sample())
    assert list(features.columns) == MODEL_FEATURE_COLUMNS
    assert features.loc[0, "color_u_g"] == 1.0
    assert features.loc[0, "color_i_z"] == 0.5


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, "not-a-number", -9999])
def test_prepare_inference_marks_invalid_rows(bad_value):
    data = sample().astype(object); data.loc[0, "modelMag_r"] = bad_value
    _, valid, reasons = prepare_inference(data)
    assert not valid.iloc[0]
    assert reasons.iloc[0]


def test_missing_column_is_rejected():
    with pytest.raises(ValueError, match="missing required"):
        engineer_features(sample().drop(columns="modelMag_z"))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        prepare_inference(sample().iloc[0:0])
