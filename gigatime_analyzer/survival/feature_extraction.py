import numpy as np
import pandas as pd


def extract_features_from_tensor(output_tensor):
    """
    Convert GigaTIME output tensor → marker feature vector.

    Parameters:
        output_tensor (torch.Tensor): shape (B, 23, H, W)

    Returns:
        np.array: shape (21,) marker intensities
    """
    output = output_tensor.detach().cpu().numpy()

    # First 21 channels = biological markers
    marker_maps = output[:, :21, :, :]

    # Average across batch + spatial dimensions
    marker_values = marker_maps.mean(axis=(0, 2, 3))

    return marker_values


def tile_name_to_patient_id(tile_name):
    """
    Convert tile name → patient ID.

    Example:
        TCGA-2J-AAB1-01Z-00-DX1 → TCGA-2J-AAB1
    """
    if isinstance(tile_name, (list, tuple)):
        tile_name = tile_name[0]

    return "-".join(tile_name.split("-")[:3])


def build_patient_dataframe(outputs, ids):
    """
    Convert model outputs → patient-level dataframe

    Parameters:
        outputs: list of tensors (one per tile)
        ids: list of tile names

    Returns:
        pd.DataFrame
    """
    patient_data = {}

    for output, tile_id in zip(outputs, ids):
        patient_id = tile_name_to_patient_id(tile_id)

        features = extract_features_from_tensor(output)

        if patient_id not in patient_data:
            patient_data[patient_id] = []

        patient_data[patient_id].append(features)

    # Aggregate tiles → patient (mean)
    rows = []
    for patient_id, feature_list in patient_data.items():
        mean_features = np.mean(feature_list, axis=0)

        row = {"patient_id": patient_id}
        for i, val in enumerate(mean_features):
            row[f"marker_{i+1}"] = val

        rows.append(row)

    return pd.DataFrame(rows)


def add_mock_survival_data(df):
    """
    TEMP: Add fake survival data for testing.
    Replace later with real TCGA clinical data.
    """
    np.random.seed(42)

    df["time"] = np.random.randint(100, 1000, size=len(df))
    df["event"] = np.random.randint(0, 2, size=len(df))

    return df