"""Helper functions for output formatting."""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

# Don't show warnings about downcasting in future pandas versions
# Warning would come from pd.replace()
pd.set_option("future.no_silent_downcasting", True)


def get_non_energy_based_reduced_features(
    df: pd.DataFrame, exclude_cols: List[str], feature_type: str, _namespace: str, _loc: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get the reduced features of a conformer ensemble that are not based on the conformer
    energies (mean, min, and max values across all valid conformers).

    Feature columns that are not numeric are excluded, and a warning is logged.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame containing the data for the individual conformers.
    exclude_cols : List[str]
        The names of the columns to exclude during the calculation of the reduced features.
    feature_type : str
        The type of features, either "atom" or "bond". This is only used for logging purposes.
    _namespace : str
        The namespace of the currently handled molecule for logging purposes.
    _loc : str
        The name of the current function for logging purposes.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing the mean, min, and max feature pandas DataFrames.
    """
    mean_features = {}
    min_features = {}
    max_features = {}

    # Loop over DataFrame with multiple conformers and group by atom/bond index
    for idx, sub_df in df.groupby(level=0):
        # Drop unnecessary columns
        sub_df_ = sub_df.drop(exclude_cols, axis=1, errors="ignore")

        # Change the dtype of each column to numeric if possible. This is necessary because some
        # columns might have been inferred as object dtype even though they contain numeric values
        # only.
        for col in sub_df_.columns:
            try:
                sub_df_[col] = pd.to_numeric(sub_df_[col])
            except ValueError:
                pass

        # Check for None values and log a warning if found
        for col in sub_df_.columns:
            if sub_df_[col].isnull().any():
                logging.warning(
                    f"'{_namespace}' | {_loc}()\nThe '{col}' feature contains None values for "
                    f"{feature_type} with index {idx}. These will be ignored during the "
                    "calculation of the mean, min, and max values. Ensure this matches your "
                    "expectations for these values. Check the unreduced features for details."
                )

        # Calculate mean, min, and max for numeric-only columns
        sub_mean = sub_df_.mean(numeric_only=True)
        sub_min = sub_df_.min(numeric_only=True)
        sub_max = sub_df_.max(numeric_only=True)

        # Log if columns could not be converted to numeric
        for col in list(sub_df_.columns):
            if col not in list(sub_mean.index):
                logging.warning(
                    f"'{_namespace}' | {_loc}()\nThe mean, min, and max value of the '{col}' "
                    f"feature could not be calculated for {feature_type} with index {idx} because "
                    "it contains non-numeric values. Check the unreduced features for details."
                )

        mean_features[idx] = sub_mean
        min_features[idx] = sub_min
        max_features[idx] = sub_max

    # Format data as DataFrames
    mean_features_df = pd.DataFrame(mean_features).T
    mean_features_df.columns = [f"MEAN__{col}" for col in mean_features_df.columns]

    min_features_df = pd.DataFrame(min_features).T
    min_features_df.columns = [f"MIN__{col}" for col in min_features_df.columns]

    max_features_df = pd.DataFrame(max_features).T
    max_features_df.columns = [f"MAX__{col}" for col in max_features_df.columns]

    return mean_features_df, min_features_df, max_features_df


def get_energy_based_reduced_features(
    df: pd.DataFrame, exclude_cols: List[str], feature_type: str, _namespace: str, _loc: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get the reduced features of a conformer ensemble that are based on the conformer energies
    (features of the lowest- and highest-energy conformer and Boltzmann-weighted features).

    If there are degenerate conformers which happen to be the lowest/highest-energy conformers, the
    minE/maxE conformer feature values of all degenerate conformers are returned and a warning is
    logged. Feature columns that are not numeric are excluded during Boltzmann weighing, and a
    warning is logged.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame containing the data for the individual conformers.
    exclude_cols : List[str]
        The names of the columns to exclude during the calculation of the reduced features.
    feature_type : str
        The type of features, either "atom" or "bond". This is only used for logging purposes.
    _namespace : str
        The namespace of the currently handled molecule for logging purposes.
    _loc : str
        The name of the current function for logging purposes.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing the pandas DataFrames for the features of the lowest-energy conformer,
        highest-energy conformer, and the Boltzmann-weighted features.
    """
    _index_name = df.index.name

    min_e_features = pd.DataFrame()
    max_e_features = pd.DataFrame()
    boltzmann_features = {}

    # Loop over DataFrame with multiple conformers and group by atom/bond index
    for idx, sub_df in df.groupby(level=0):
        # Calculate features of the min energy and max energy conformers
        min_e = sub_df.loc[sub_df["conformer_energy"] == sub_df["conformer_energy"].min()]
        max_e = sub_df.loc[sub_df["conformer_energy"] == sub_df["conformer_energy"].max()]

        min_e = min_e.drop(exclude_cols, axis=1, errors="ignore")
        max_e = max_e.drop(exclude_cols, axis=1, errors="ignore")

        # Change "_inaccessible" to NaN
        min_e = min_e.replace("_inaccessible", np.nan)
        min_e = min_e.infer_objects(copy=False)
        max_e = max_e.replace("_inaccessible", np.nan)
        max_e = max_e.infer_objects(copy=False)

        # Warn if multiple conformers have the same min/max energy
        if len(min_e) > 1:
            logging.warning(
                f"'{_namespace}' | {_loc}()\nMultiple conformers have the same lowest energy "
                f"for {feature_type} with index {idx}. The features of all these conformers are "
                "returned in the min energy features DataFrame. Ensure this matches your "
                "expectations for these values. Check the unreduced features for details."
            )
        if len(max_e) > 1:
            logging.warning(
                f"'{_namespace}' | {_loc}()\nMultiple conformers have the same highest energy "
                f"for {feature_type} with index {idx}. The features of all these conformers are "
                "returned in the max energy features DataFrame. Ensure this matches your "
                "expectations for these values. Check the unreduced features for details."
            )

        # Drop conformers (rows) that don't have a Boltzmann weight assigned and get weights
        sub_df_ = sub_df.dropna(subset="boltzmann_weight")
        weights = sub_df_["boltzmann_weight"]

        # Prepare DataFrame for Boltzmann-weighted average calculation
        sub_df_ = sub_df_.drop(exclude_cols, axis=1, errors="ignore")

        # Ignore non-numeric columns by setting them to None
        for col in sub_df_.columns:
            try:
                sub_df_[col] = pd.to_numeric(sub_df_[col])
            except ValueError:
                logging.warning(
                    f"'{_namespace}' | {_loc}()\nThe Boltzmann-weighted average value of the "
                    f"'{col}' feature could not be calculated for {feature_type} with index {idx} "
                    "because it contains non-numeric values. Check the unreduced features "
                    "for details."
                )
                sub_df_[col] = None

        # Calculate Boltzmann-weighted average for each feature -
        boltzmann = sub_df_.apply(
            lambda col: None
            if col.isna().all() is np.True_
            else (col * weights).sum() / weights.sum()
        )

        # Append reduced features to overall DataFrames
        min_e_features = pd.concat([min_e_features, min_e])
        max_e_features = pd.concat([max_e_features, max_e])
        boltzmann_features[idx] = boltzmann

    # Format min_e, max_e, and Boltzmann features DataFrames
    min_e_features.columns = [f"LOWEST_ENERGY__{col}" for col in min_e_features.columns]
    max_e_features.columns = [f"HIGHEST_ENERGY__{col}" for col in max_e_features.columns]

    boltzmann_features_df = pd.DataFrame(boltzmann_features).T
    boltzmann_features_df.columns = [f"BOLTZMANN__{col}" for col in boltzmann_features_df.columns]
    boltzmann_features_df.index.name = _index_name

    return min_e_features, max_e_features, boltzmann_features_df
