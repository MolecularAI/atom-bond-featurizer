"""Utility methods for analyzing log files from BONAFIDE after or during feature generation."""

from datetime import datetime
from typing import List, Optional

import pandas as pd


class LogFileAnalyzer:
    """Analyze a log file from the Bond and Atom Featurizer and Descriptor Extractor (BONAFIDE).

    Parameters
    ----------
    log_file_path : str
        The path to the log file to analyze.

    Attributes
    ----------
    log_file_lines : List[str]
        A list of the lines of the log file.
    """

    def __init__(self, log_file_path: str) -> None:
        self.log_file_path: str = log_file_path
        self.log_file_lines: List[str] = []

        self._read_file()

    def _read_file(self) -> None:
        """Read the log file.

        Returns
        -------
        None
        """
        try:
            with open(self.log_file_path, "r") as f:
                self.log_file_lines = f.readlines()
        except Exception as e:
            raise IOError(f"Error reading log file: {e}")

        if self.log_file_lines:
            while self.log_file_lines and self.log_file_lines[-1].strip() == "":
                self.log_file_lines.pop()

    def _get_time_stamp(self, time_string: str) -> datetime:
        """Convert a time string to a datetime object.

        Parameters
        ----------
        time_string : str
            The time string to convert, expected format: "YYYY-MM-DD HH:MM:SS".

        Returns
        -------
        datetime
            The corresponding datetime object if the conversion was successful.
        """
        try:
            return datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            raise ValueError(f"Error parsing time string: {e}")

    def get_level_log_messages(self, log_level: str = "ERROR") -> str:
        """Get all log messages of a specific logging level.

        Parameters
        ----------
        log_level : str, optional
            The desired logging level, by default "ERROR".

        Returns
        -------
        str
            A string containing all log messages of the specified logging level, including
            any indented lines that follow each log message.
        """
        log_level = log_level.upper()

        all_to_be_returned = ""
        for line_idx, line in enumerate(self.log_file_lines):
            if f"| {log_level} |" in line:
                to_be_returned = [line]

                for line2 in self.log_file_lines[line_idx + 1 :]:
                    if line2.startswith(" "):
                        to_be_returned.append(line2)
                    else:
                        break

                all_to_be_returned += "".join(to_be_returned)

        return all_to_be_returned

    def check_string_in_last_line(self, target_string: str) -> bool:
        """Check if a specific string is present in the last line of the log file.

        Parameters
        ----------
        target_string : str
            The string to check for in the last line of the log file.

        Returns
        -------
        bool
            ``True`` if the target string is found in the last line, ``False`` otherwise.
        """
        return target_string in self.log_file_lines[-1]

    def get_time_for_individual_features(self) -> pd.DataFrame:
        """Get the elapsed time for each individual feature.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names as index and columns for elapsed time, start time,
            end time, and feature type.
        """
        time_dict = {}

        for line_idx, line in enumerate(self.log_file_lines):
            if "Validated configuration settings: {" in line:
                start_time = self._get_time_stamp(
                    self.log_file_lines[line_idx - 1].split("|")[0].strip()
                )

                feature_name = None
                for line_idx2, line2 in enumerate(self.log_file_lines[line_idx:]):
                    if (
                        "-atom-" in line2 or "-bond-" in line2
                    ) and "configuration settings:" not in line2:
                        feature_name = line2.split("'")[1]
                        break

                if feature_name is None:
                    continue

                lines_inverted = self.log_file_lines[::-1]
                for line_idx3, line3 in enumerate(lines_inverted):
                    if f"'{feature_name}'" in line3 and "configuration settings:" not in line3:
                        if (
                            "AtomBondFeaturizer.featurize_atoms()" in lines_inverted[line_idx3 + 1]
                            or "AtomBondFeaturizer.featurize_bonds()"
                            in lines_inverted[line_idx3 + 1]
                        ):
                            end_time = self._get_time_stamp(
                                lines_inverted[line_idx3 + 1].split("|")[0].strip()
                            )

                        elif (
                            "AtomBondFeaturizer.featurize_atoms()" in lines_inverted[line_idx3 + 2]
                            or "AtomBondFeaturizer.featurize_bonds()"
                            in lines_inverted[line_idx3 + 2]
                        ):
                            end_time = self._get_time_stamp(
                                lines_inverted[line_idx3 + 2].split("|")[0].strip()
                            )
                        else:
                            raise ValueError("Could not find end time for feature.")

                        break

                elapsed_time = end_time - start_time
                time_dict[feature_name] = {
                    "elapsed_time [s]": elapsed_time.total_seconds(),
                    "start_time": start_time.strftime("%H:%M:%S"),
                    "end_time": end_time.strftime("%H:%M:%S"),
                    "feature_type": "atom" if "-atom-" in feature_name else "bond",
                }

        df = pd.DataFrame(time_dict).T.sort_values("elapsed_time [s]", ascending=False)
        df["elapsed_time [s]"] = df["elapsed_time [s]"].astype(int)
        return df

    def get_total_time_for_atom_featurization(self) -> float:
        """Get the total time taken for atom featurization.

        Returns
        -------
        float
            The total time taken for atom featurization in seconds.
        """
        start_time: Optional[datetime] = None
        end_time: Optional[datetime] = None

        for line in self.log_file_lines:
            if "| AtomBondFeaturizer.featurize_atoms() | START" in line:
                try:
                    start_time = self._get_time_stamp(line.split("|")[0].strip())
                except Exception as e:
                    raise ValueError(f"Error parsing start time for atom featurization: {e}")

            if "| AtomBondFeaturizer.featurize_atoms() | DONE" in line:
                try:
                    end_time = self._get_time_stamp(line.split("|")[0].strip())
                except Exception as e:
                    raise ValueError(f"Error parsing end time for atom featurization: {e}")

        if start_time is None or end_time is None:
            raise ValueError(
                "Could not find start and/or end time for atom featurization in log file."
            )

        return (end_time - start_time).total_seconds()

    def get_total_time_for_bond_featurization(self) -> float:
        """Get the total time taken for bond featurization.

        Returns
        -------
        float
            The total time taken for bond featurization in seconds.
        """
        start_time: Optional[datetime] = None
        end_time: Optional[datetime] = None

        for line in self.log_file_lines:
            if "| AtomBondFeaturizer.featurize_bonds() | START" in line:
                try:
                    start_time = self._get_time_stamp(line.split("|")[0].strip())
                except Exception as e:
                    raise ValueError(f"Error parsing start time for bond featurization: {e}")

            if "| AtomBondFeaturizer.featurize_bonds() | DONE" in line:
                try:
                    end_time = self._get_time_stamp(line.split("|")[0].strip())
                except Exception as e:
                    raise ValueError(f"Error parsing end time for bond featurization: {e}")

        if start_time is None or end_time is None:
            raise ValueError(
                "Could not find start and/or end time for bond featurization in log file."
            )

        return (end_time - start_time).total_seconds()

    def get_total_runtime(self) -> float:
        """Get the total runtime.

        Returns
        -------
        float
            The total runtime in seconds.
        """
        if not self.log_file_lines:
            raise ValueError("Log file is empty.")

        first_line = self.log_file_lines[0]

        last_line = None
        for line in self.log_file_lines[::-1]:
            if not line.startswith(" "):
                last_line = line
                break

        if last_line is None:
            raise ValueError("Could not find a valid last line in log file.")

        try:
            start_time = datetime.strptime(first_line.split("|")[0].strip(), "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            raise ValueError(f"Error parsing start time from log file: {e}")

        try:
            end_time = self._get_time_stamp(last_line.split("|")[0].strip())
        except Exception as e:
            raise ValueError(f"Error parsing end time from log file: {e}")

        return (end_time - start_time).total_seconds()
