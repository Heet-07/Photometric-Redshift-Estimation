import json

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.predict import load_artifacts, predict_dataframe, write_predictions


def make_artifacts(tmp_path):
    features = ["modelMag_u", "modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z", "color_u_g", "color_g_r", "color_r_i", "color_i_z"]
    model = xgb.XGBRegressor(n_estimators=2, max_depth=2, random_state=42, objective="reg:squarederror")
    X = np.tile(np.arange(9, dtype=float), (4, 1)); y = np.array([.05, .1, .15, .2])
    model.fit(X, y)
    model_path = tmp_path / "model.json"; metadata_path = tmp_path / "metadata.json"
    model.save_model(model_path); metadata_path.write_text(json.dumps({"feature_columns": features, "target_column": "specz"}), encoding="utf-8")
    return model_path, metadata_path


def input_frame():
    return pd.DataFrame({"objID": [10, 11], "modelMag_u": [20., 20.], "modelMag_g": [19., 19.], "modelMag_r": [18., -9999.], "modelMag_i": [17., 17.], "modelMag_z": [16., 16.]})


def test_prediction_preserves_rows_and_marks_invalid(tmp_path):
    model_path, metadata_path = make_artifacts(tmp_path)
    model, _ = load_artifacts("evaluation", model_path, metadata_path)
    output = predict_dataframe(input_frame(), model)
    assert list(output["objID"]) == [10, 11]
    assert output.loc[0, "prediction_status"] == "predicted"
    assert output.loc[1, "prediction_status"] == "invalid"
    assert np.isfinite(output.loc[0, "predicted_specz"])


def test_metadata_mismatch_is_rejected(tmp_path):
    model_path, metadata_path = make_artifacts(tmp_path)
    metadata_path.write_text(json.dumps({"feature_columns": ["wrong"], "target_column": "specz"}), encoding="utf-8")
    with pytest.raises(ValueError, match="feature schema"):
        load_artifacts("evaluation", model_path, metadata_path)


def test_write_predictions_creates_output(tmp_path):
    path = tmp_path / "predictions.csv"
    write_predictions(pd.DataFrame({"predicted_specz": [.1]}), path)
    assert path.is_file()
