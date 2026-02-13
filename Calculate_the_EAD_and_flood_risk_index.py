#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Flood Hazard EAD (Expected Annual Damage) Calculator
================================================================================

A comprehensive GUI application for calculating Expected Annual Damage (EAD)
and composite Flood Risk Index from large-scale parquet datasets.

Features:
- Modern GUI with CustomTkinter
- Memory-efficient chunked processing for large datasets (30-40GB+)
- Dynamic field selection: user analyzes input file and manually selects
  which columns represent water depth at different return periods
- EAD calculation using trapezoidal integration
- Composite Risk Index calculation
- Progress monitoring and detailed logging

Author: LONG
Date: 2026
Version: 2.0.0

References:
1. Meyer, V., et al. (2009). IEAM, 5(1), 17-26.
2. De Moel, H., et al. (2015). MASGC, 20(6), 865-890.
3. Wing, O.E.J., et al. (2022). Nature Climate Change, 12, 156-162.
================================================================================
"""

import os
import sys
import time
import gc
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import warnings

# Third-party imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# GUI imports
try:
    import customtkinter as ctk
    from customtkinter import CTk, CTkFrame, CTkLabel, CTkButton, CTkEntry
    from customtkinter import CTkTextbox, CTkProgressBar, CTkScrollableFrame
    from customtkinter import CTkOptionMenu, CTkCheckBox
except ImportError:
    print("Installing customtkinter...")
    os.system("pip install customtkinter --break-system-packages")
    import customtkinter as ctk
    from customtkinter import CTk, CTkFrame, CTkLabel, CTkButton, CTkEntry
    from customtkinter import CTkTextbox, CTkProgressBar, CTkScrollableFrame
    from customtkinter import CTkOptionMenu, CTkCheckBox

from tkinter import filedialog, messagebox
import tkinter as tk

# Suppress warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================

# Application Configuration
APP_TITLE = "Flood Hazard EAD Calculator v2.0"
APP_WIDTH = 1400
APP_HEIGHT = 950
THEME_MODE = "dark"
COLOR_THEME = "blue"

# Processing Configuration
CHUNK_SIZE = 500_000       # Rows per chunk for memory efficiency
MAX_MEMORY_PERCENT = 75    # Maximum memory usage percentage
NUM_WORKERS = 8            # Number of parallel workers (adjust based on CPU)

# Default Return Periods (user can customize via GUI)
DEFAULT_RETURN_PERIODS = [10, 20, 50, 75, 100, 200, 500]

# Default Paths
DEFAULT_INPUT_PATH = r"E:\Global_Urban_age\FIG\DATA\Urban_age_V17.parquet"
DEFAULT_OUTPUT_DIR = r"E:\Global_Urban_age\FIG\DATA"
DEFAULT_OUTPUT_FILE = "Built_Up_age_V17.parquet"

# Color Scheme
COLORS = {
    'primary': '#1E88E5',
    'secondary': '#424242',
    'success': '#4CAF50',
    'warning': '#FFC107',
    'danger': '#F44336',
    'info': '#00BCD4',
    'bg_dark': '#1a1a2e',
    'bg_medium': '#16213e',
    'bg_light': '#0f3460',
    'text_light': '#e8e8e8',
    'text_muted': '#a0a0a0'
}

# Placeholder for "not selected" in dropdown menus
_NONE_PLACEHOLDER = "-- Not Selected --"


# ==============================================================================
# EAD CALCULATION ENGINE
# ==============================================================================

class EADCalculator:
    """
    Expected Annual Damage (EAD) Calculator using Trapezoidal Integration.

    The EAD is calculated by integrating flood depths over their probability
    of occurrence using the trapezoidal rule.

    Formula: EAD_i = integral_0^1 H_i(p) dp

    Where:
    - p is the Annual Exceedance Probability (AEP = 1/T)
    - H_i(p) is the normalized flood depth at grid cell i
    """

    def __init__(self, return_periods: List[float]):
        """
        Initialize the EAD Calculator.

        Args:
            return_periods: List of return periods in years,
                            e.g. [10, 20, 50, 100, 200, 500]
        """
        self.return_periods = np.array(return_periods, dtype=np.float64)
        self.exceedance_probs = 1.0 / self.return_periods

        # Pre-compute boundary-extended probabilities sorted ascending
        sort_idx = np.argsort(self.exceedance_probs)
        self.sorted_probs = self.exceedance_probs[sort_idx]
        self.sort_indices = sort_idx

        # Extended probability array with boundaries [0, p1, ..., pn, 1]
        self.p_extended = np.concatenate([[0], self.sorted_probs, [1]])

        # Pre-compute delta_p for integration
        self.delta_p = np.diff(self.p_extended)

        # Compute EAD_max for normalization (when all H = 1)
        H_max = np.ones(len(self.return_periods))
        H_sorted = H_max[sort_idx]
        H_extended = np.concatenate([[H_sorted[0]], H_sorted, [H_sorted[-1]]])
        avg_H = (H_extended[:-1] + H_extended[1:]) / 2
        self.ead_max = np.sum(avg_H * self.delta_p)

    def calculate_ead_vectorized(self, hazard_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate EAD for multiple grid cells (vectorized).

        Args:
            hazard_matrix: 2D array of shape (n_cells, n_return_periods).
                           Columns must match the order of self.return_periods.

        Returns:
            1D array of normalized EAD values (0-1 range).
        """
        # Sort by probability (same order for all cells)
        H_sorted = hazard_matrix[:, self.sort_indices]

        # Add boundary conditions
        H_extended = np.column_stack([
            H_sorted[:, 0],   # Boundary at p=0 (extend from longest RP)
            H_sorted,          # Original sorted values
            H_sorted[:, -1]    # Boundary at p=1 (extend from shortest RP)
        ])

        # Trapezoidal integration
        avg_H = (H_extended[:, :-1] + H_extended[:, 1:]) / 2
        ead_values = np.sum(avg_H * self.delta_p, axis=1)

        # Normalize and clip to [0, 1]
        normalized_ead = np.clip(ead_values / self.ead_max, 0.0, 1.0)

        return normalized_ead.astype(np.float32)

    def calculate_risk_index(self,
                             hazard: np.ndarray,
                             exposure: np.ndarray,
                             vulnerability: np.ndarray) -> np.ndarray:
        """
        Calculate Composite Flood Risk Index.

        Formula: Risk_i = (Hazard_i * Exposure_i * Vulnerability_i)^(1/3)
        """
        hazard = np.clip(np.nan_to_num(hazard, nan=0.0), 0.0, 1.0)
        exposure = np.clip(np.nan_to_num(exposure, nan=0.0), 0.0, 1.0)
        vulnerability = np.clip(np.nan_to_num(vulnerability, nan=0.0), 0.0, 1.0)

        product = hazard * exposure * vulnerability
        risk_index = np.power(product, 1.0 / 3.0)

        return risk_index.astype(np.float32)


# ==============================================================================
# DATA PROCESSOR (Memory-Efficient Chunked Processing)
# ==============================================================================

class DataProcessor:
    """
    Memory-efficient processor for large Parquet datasets.

    Accepts user-selected hazard columns and their corresponding return
    periods rather than relying on hardcoded column names.
    """

    def __init__(self,
                 input_path: str,
                 output_path: str,
                 hazard_columns: List[str],
                 return_periods: List[float],
                 exposure_col: str = 'Exposure_Index',
                 vulnerability_col: str = 'Vulnerability_Index',
                 chunk_size: int = CHUNK_SIZE,
                 log_callback=None,
                 progress_callback=None):
        """
        Initialize the Data Processor.

        Args:
            input_path:        Path to input Parquet file
            output_path:       Path to output Parquet file
            hazard_columns:    Column names containing water-depth values
            return_periods:    Corresponding return periods (same order)
            exposure_col:      Exposure index column name
            vulnerability_col: Vulnerability index column name
            chunk_size:        Rows per processing chunk
            log_callback:      Logging function
            progress_callback: Progress update function
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.hazard_columns = hazard_columns
        self.return_periods = return_periods
        self.exposure_col = exposure_col
        self.vulnerability_col = vulnerability_col
        self.chunk_size = chunk_size
        self.log = log_callback or print
        self.progress = progress_callback or (lambda x, y: None)

        self.ead_calculator = EADCalculator(return_periods=self.return_periods)
        self._stop_flag = False
        self._output_schema = None

    def stop(self):
        """Signal the processor to stop."""
        self._stop_flag = True

    def reset(self):
        """Reset the stop flag."""
        self._stop_flag = False

    def _log_memory_usage(self):
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            mem_gb = process.memory_info().rss / (1024 ** 3)
            self.log(f"[MEMORY] Current usage: {mem_gb:.2f} GB")
        except ImportError:
            pass

    def _create_output_schema(self, input_schema):
        """Create a consistent output schema based on the input schema."""
        nullable_int_columns = {
            'Income_Classification', 'Country', 'Sovereig', 'Developed'
        }

        new_fields = []
        for field in input_schema:
            if field.name in nullable_int_columns:
                new_fields.append(pa.field(field.name, pa.float64()))
            else:
                new_fields.append(field)

        # Add new calculated columns
        new_fields.append(pa.field('Hazard_Index_EAD', pa.float32()))
        new_fields.append(pa.field('Risk_Index', pa.float32()))

        return pa.schema(new_fields)

    def _ensure_schema_consistency(self, df, schema):
        """Ensure DataFrame matches the expected schema."""
        for field in schema:
            col_name = field.name
            if col_name not in df.columns:
                continue

            pa_type = field.type
            if pa.types.is_float64(pa_type):
                df[col_name] = df[col_name].astype('float64')
            elif pa.types.is_float32(pa_type):
                df[col_name] = df[col_name].astype('float32')
            elif pa.types.is_int64(pa_type):
                if df[col_name].isna().any():
                    df[col_name] = df[col_name].astype('float64')
                else:
                    df[col_name] = df[col_name].astype('int64')
            elif pa.types.is_int32(pa_type):
                if df[col_name].isna().any():
                    df[col_name] = df[col_name].astype('float64')
                else:
                    df[col_name] = df[col_name].astype('int32')
            elif pa.types.is_int16(pa_type):
                if df[col_name].isna().any():
                    df[col_name] = df[col_name].astype('float64')
                else:
                    df[col_name] = df[col_name].astype('int16')

        return df

    def validate_input(self) -> Tuple[bool, str, Dict]:
        """Validate the input Parquet file."""
        self.log("[VALIDATE] Validating input file...")

        if not self.input_path.exists():
            return False, f"Input file not found: {self.input_path}", {}

        try:
            parquet_file = pq.ParquetFile(self.input_path)
            metadata = parquet_file.metadata
            schema = parquet_file.schema_arrow

            num_rows = metadata.num_rows
            num_columns = metadata.num_columns
            file_size_gb = self.input_path.stat().st_size / (1024 ** 3)

            self.log(f"[VALIDATE] File size: {file_size_gb:.2f} GB")
            self.log(f"[VALIDATE] Total rows: {num_rows:,}")
            self.log(f"[VALIDATE] Total columns: {num_columns}")

            column_names = [field.name for field in schema]

            # Check all required columns exist
            missing_columns = []
            for col in self.hazard_columns:
                if col not in column_names:
                    missing_columns.append(col)
            for col in [self.exposure_col, self.vulnerability_col]:
                if col and col not in column_names:
                    missing_columns.append(col)

            if missing_columns:
                return False, f"Missing required columns: {missing_columns}", {}

            self.log("[VALIDATE] All required columns found!")

            num_chunks = (num_rows + self.chunk_size - 1) // self.chunk_size
            self.log(f"[VALIDATE] Estimated chunks: {num_chunks}")

            meta_dict = {
                'num_rows': num_rows,
                'num_columns': num_columns,
                'file_size_gb': file_size_gb,
                'num_chunks': num_chunks,
                'columns': column_names
            }

            return True, "Validation successful!", meta_dict

        except Exception as e:
            return False, f"Validation error: {str(e)}", {}

    def process(self) -> Tuple[bool, str]:
        """Process the entire Parquet file in chunks."""
        start_time = time.time()
        self.log("=" * 70)
        self.log("[PROCESS] Starting EAD and Risk Index calculation...")
        self.log(f"[PROCESS] Water depth columns ({len(self.hazard_columns)}):")
        for col, rp in zip(self.hazard_columns, self.return_periods):
            self.log(f"          {col}  ->  RP = {rp} yr")
        self.log("=" * 70)

        # Validate input
        is_valid, msg, meta = self.validate_input()
        if not is_valid:
            return False, msg

        num_rows = meta['num_rows']
        num_chunks = meta['num_chunks']

        try:
            parquet_file = pq.ParquetFile(self.input_path)
            input_schema = parquet_file.schema_arrow

            self.log("[PROCESS] Creating consistent output schema...")
            self._output_schema = self._create_output_schema(input_schema)
            self.log(f"[PROCESS] Output schema has {len(self._output_schema)} fields")

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_output = self.output_path.with_suffix('.tmp.parquet')

            writer = None
            processed_rows = 0
            chunk_idx = 0

            self.log(f"[PROCESS] Starting chunked processing...")
            self.log(f"[PROCESS] Chunk size: {self.chunk_size:,} rows")

            for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                if self._stop_flag:
                    self.log("[PROCESS] Stopping by user request...")
                    if writer:
                        writer.close()
                    if temp_output.exists():
                        temp_output.unlink()
                    return False, "Processing stopped by user"

                chunk_start_time = time.time()
                chunk_idx += 1

                df = batch.to_pandas()
                chunk_rows = len(df)

                self.log(f"\n[CHUNK {chunk_idx}/{num_chunks}] "
                         f"Processing {chunk_rows:,} rows...")

                # Extract user-selected hazard columns
                hazard_matrix = df[self.hazard_columns].values.astype(np.float64)
                hazard_matrix = np.nan_to_num(hazard_matrix, nan=0.0)

                # Calculate EAD
                self.log(f"[CHUNK {chunk_idx}] Calculating EAD "
                         f"(trapezoidal integration)...")
                ead_values = self.ead_calculator.calculate_ead_vectorized(
                    hazard_matrix)
                df['Hazard_Index_EAD'] = ead_values.astype(np.float32)

                # Calculate Risk Index
                self.log(f"[CHUNK {chunk_idx}] Calculating composite "
                         f"Risk Index...")
                exposure = df[self.exposure_col].values.astype(np.float64)
                vulnerability = df[self.vulnerability_col].values.astype(
                    np.float64)

                risk_values = self.ead_calculator.calculate_risk_index(
                    ead_values, exposure, vulnerability
                )
                df['Risk_Index'] = risk_values.astype(np.float32)

                # Ensure schema consistency
                self.log(f"[CHUNK {chunk_idx}] Ensuring schema consistency...")
                df = self._ensure_schema_consistency(df, self._output_schema)

                schema_columns = [f.name for f in self._output_schema]
                df_filtered = df[
                    [c for c in schema_columns if c in df.columns]]

                table = pa.Table.from_pandas(
                    df_filtered, schema=self._output_schema,
                    preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(
                        str(temp_output), self._output_schema,
                        compression='snappy')

                writer.write_table(table)

                processed_rows += chunk_rows
                progress_pct = processed_rows / num_rows
                self.progress(
                    progress_pct,
                    f"Processed {processed_rows:,}/{num_rows:,} rows")

                chunk_time = time.time() - chunk_start_time
                rows_per_sec = chunk_rows / chunk_time
                self.log(f"[CHUNK {chunk_idx}] Done in {chunk_time:.2f}s "
                         f"({rows_per_sec:,.0f} rows/sec)")

                del df, df_filtered, hazard_matrix
                del ead_values, risk_values, table
                gc.collect()

                if chunk_idx % 10 == 0:
                    self._log_memory_usage()

            if writer:
                writer.close()

            if temp_output.exists():
                if self.output_path.exists():
                    self.output_path.unlink()
                temp_output.rename(self.output_path)

            total_time = time.time() - start_time
            self.log("\n" + "=" * 70)
            self.log("[COMPLETE] Processing finished successfully!")
            self.log(f"[COMPLETE] Total rows: {processed_rows:,}")
            self.log(f"[COMPLETE] Total time: {total_time:.2f} seconds")
            self.log(f"[COMPLETE] Avg speed: "
                     f"{processed_rows / total_time:,.0f} rows/sec")
            self.log(f"[COMPLETE] Output: {self.output_path}")
            self.log("=" * 70)

            return True, "Processing completed successfully!"

        except Exception as e:
            self.log(f"[ERROR] Processing failed: {str(e)}")
            import traceback
            self.log(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            return False, f"Processing error: {str(e)}"


# ==============================================================================
# FIELD MAPPING DIALOG (replaces hardcoded column detection)
# ==============================================================================

class FieldMappingDialog(ctk.CTkToplevel):
    """
    Dialog window that lets the user map file columns to return-period
    water-depth fields.  Users can add/remove return-period rows freely.
    """

    def __init__(self, parent, all_columns: List[str],
                 current_mappings: List[Tuple[float, str]] = None):
        """
        Args:
            parent:           Parent widget.
            all_columns:      All column names from the parquet file.
            current_mappings: Pre-existing [(return_period, column), ...].
        """
        super().__init__(parent)
        self.title("Field Mapping - Select Water Depth Columns")
        self.geometry("750x700")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        self.all_columns = sorted(all_columns)
        self.column_options = [_NONE_PLACEHOLDER] + self.all_columns
        self.result: Optional[List[Tuple[float, str]]] = None

        self.exposure_result: Optional[str] = None
        self.vulnerability_result: Optional[str] = None

        # Mapping rows: list of (rp_var, col_var)
        self.mapping_rows: List[Tuple[tk.StringVar, tk.StringVar]] = []

        self._build_ui(current_mappings)

    # ------------------------------------------------------------------
    def _build_ui(self, current_mappings):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Header
        header = CTkLabel(
            self,
            text="Select the water depth column for each return period",
            font=ctk.CTkFont(size=16, weight="bold"))
        header.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="w")

        hint = CTkLabel(
            self,
            text=("Tip: Click 'Add Return Period' to add rows. "
                  "Set a field to '-- Not Selected --' to skip it."),
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_muted'])
        hint.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

        # Scrollable mapping area
        self.scroll_frame = CTkScrollableFrame(
            self,
            label_text="Water Depth Field Mapping "
                       "(Return Period -> Column)")
        self.scroll_frame.grid(row=2, column=0, padx=15, pady=5,
                               sticky="nsew")
        self.scroll_frame.grid_columnconfigure(1, weight=1)

        # Column headers
        CTkLabel(self.scroll_frame, text="Return Period (yr)",
                 font=ctk.CTkFont(size=13, weight="bold")
                 ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        CTkLabel(self.scroll_frame, text="Water Depth Column",
                 font=ctk.CTkFont(size=13, weight="bold")
                 ).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        CTkLabel(self.scroll_frame, text="Action",
                 font=ctk.CTkFont(size=13, weight="bold")
                 ).grid(row=0, column=2, padx=5, pady=5)

        # Populate initial rows
        if current_mappings:
            for rp, col in current_mappings:
                self._add_mapping_row(rp_value=str(int(rp)),
                                      col_value=col)
        else:
            for rp in DEFAULT_RETURN_PERIODS:
                self._add_mapping_row(rp_value=str(rp),
                                      col_value=_NONE_PLACEHOLDER)

        # Add-row button
        add_btn = CTkButton(
            self, text="+ Add Return Period", width=180, height=32,
            command=lambda: self._add_mapping_row())
        add_btn.grid(row=3, column=0, padx=20, pady=(8, 5), sticky="w")

        # Exposure / Vulnerability selectors
        ev_frame = CTkFrame(self, fg_color="transparent")
        ev_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        ev_frame.grid_columnconfigure(1, weight=1)

        CTkLabel(ev_frame, text="Exposure Column:",
                 font=ctk.CTkFont(size=13, weight="bold")
                 ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.exposure_var = tk.StringVar(value=_NONE_PLACEHOLDER)
        self.exposure_menu = CTkOptionMenu(
            ev_frame, variable=self.exposure_var,
            values=self.column_options, width=300)
        self.exposure_menu.grid(row=0, column=1, padx=5, pady=5,
                                sticky="w")

        CTkLabel(ev_frame, text="Vulnerability Column:",
                 font=ctk.CTkFont(size=13, weight="bold")
                 ).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.vulnerability_var = tk.StringVar(value=_NONE_PLACEHOLDER)
        self.vulnerability_menu = CTkOptionMenu(
            ev_frame, variable=self.vulnerability_var,
            values=self.column_options, width=300)
        self.vulnerability_menu.grid(row=1, column=1, padx=5, pady=5,
                                     sticky="w")

        # Auto-select if columns exist
        for col in self.all_columns:
            if col == "Exposure_Index":
                self.exposure_var.set(col)
            if col == "Vulnerability_Index":
                self.vulnerability_var.set(col)

        # OK / Cancel buttons
        btn_frame = CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=5, column=0, padx=20, pady=(5, 15), sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        ok_btn = CTkButton(
            btn_frame, text="Confirm", height=40, width=150,
            fg_color=COLORS['success'], hover_color="#388E3C",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._on_ok)
        ok_btn.grid(row=0, column=0, padx=5, sticky="e")

        cancel_btn = CTkButton(
            btn_frame, text="Cancel", height=40, width=150,
            fg_color=COLORS['danger'], hover_color="#C62828",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._on_cancel)
        cancel_btn.grid(row=0, column=1, padx=5, sticky="w")

    # ------------------------------------------------------------------
    def _add_mapping_row(self, rp_value: str = "",
                         col_value: str = _NONE_PLACEHOLDER):
        """Add one return-period to column mapping row."""
        row_idx = len(self.mapping_rows) + 1  # row 0 is header

        rp_var = tk.StringVar(value=rp_value)
        col_var = tk.StringVar(value=col_value)

        rp_entry = CTkEntry(
            self.scroll_frame, textvariable=rp_var,
            width=100, height=32, placeholder_text="e.g. 100")
        rp_entry.grid(row=row_idx, column=0, padx=5, pady=3, sticky="w")

        col_menu = CTkOptionMenu(
            self.scroll_frame, variable=col_var,
            values=self.column_options, width=350)
        col_menu.grid(row=row_idx, column=1, padx=5, pady=3, sticky="ew")

        # Auto-match column names containing the RP number
        if col_value == _NONE_PLACEHOLDER and rp_value:
            for c in self.all_columns:
                rp_padded = rp_value.zfill(3)
                if (f"RP{rp_padded}" in c or f"RP{rp_value}" in c or
                        f"rp{rp_padded}" in c or f"rp_{rp_value}" in c):
                    col_var.set(c)
                    break

        del_btn = CTkButton(
            self.scroll_frame, text="Remove", width=70, height=32,
            fg_color=COLORS['danger'], hover_color="#C62828",
            command=lambda r=row_idx: self._remove_mapping_row(r))
        del_btn.grid(row=row_idx, column=2, padx=5, pady=3)

        self.mapping_rows.append((rp_var, col_var))

    def _remove_mapping_row(self, row_idx):
        """Remove a mapping row."""
        actual_index = row_idx - 1
        if 0 <= actual_index < len(self.mapping_rows):
            for widget in self.scroll_frame.grid_slaves(row=row_idx):
                widget.destroy()
            self.mapping_rows[actual_index] = (None, None)

    # ------------------------------------------------------------------
    def _on_ok(self):
        """Collect results and close the dialog."""
        mappings = []
        for rp_var, col_var in self.mapping_rows:
            if rp_var is None:  # removed row
                continue
            rp_str = rp_var.get().strip()
            col_str = col_var.get().strip()
            if not rp_str or col_str == _NONE_PLACEHOLDER:
                continue
            try:
                rp_float = float(rp_str)
            except ValueError:
                messagebox.showerror(
                    "Input Error",
                    f"Return period '{rp_str}' is not a valid number!",
                    parent=self)
                return
            if col_str not in self.all_columns:
                messagebox.showerror(
                    "Input Error",
                    f"Column '{col_str}' does not exist in the file!",
                    parent=self)
                return
            mappings.append((rp_float, col_str))

        if len(mappings) < 2:
            messagebox.showwarning(
                "Warning",
                "At least 2 return-period water depth columns are "
                "required to calculate EAD.",
                parent=self)
            return

        # Check for duplicate column selections
        cols = [m[1] for m in mappings]
        if len(cols) != len(set(cols)):
            messagebox.showwarning(
                "Warning",
                "Duplicate column selections detected. "
                "Please check your mappings.",
                parent=self)
            return

        # Exposure / Vulnerability
        exp = self.exposure_var.get().strip()
        vul = self.vulnerability_var.get().strip()
        if exp == _NONE_PLACEHOLDER:
            messagebox.showerror(
                "Input Error",
                "Please select an Exposure column!",
                parent=self)
            return
        if vul == _NONE_PLACEHOLDER:
            messagebox.showerror(
                "Input Error",
                "Please select a Vulnerability column!",
                parent=self)
            return

        self.result = mappings
        self.exposure_result = exp
        self.vulnerability_result = vul
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()


# ==============================================================================
# GUI APPLICATION
# ==============================================================================

class FloodEADCalculatorApp(CTk):
    """
    Main GUI Application for Flood EAD Calculator.

    v2 changes:
    - Added 'Analyze Input File' button that reads the file schema
    - Opens a field-mapping dialog for user to select water-depth columns
    - Removes the hardcoded HAZARD_COLUMNS dependency
    """

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title(APP_TITLE)
        self.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.minsize(1200, 700)

        # Appearance
        ctk.set_appearance_mode(THEME_MODE)
        ctk.set_default_color_theme(COLOR_THEME)

        # Grid weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)

        # Variables
        self.input_path = tk.StringVar(value=DEFAULT_INPUT_PATH)
        self.output_dir = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        self.output_file = tk.StringVar(value=DEFAULT_OUTPUT_FILE)
        self.processor: Optional[DataProcessor] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.log_queue = queue.Queue()

        # Field mapping state
        self.field_mappings: List[Tuple[float, str]] = []
        self.exposure_col: str = "Exposure_Index"
        self.vulnerability_col: str = "Vulnerability_Index"
        self._file_columns: List[str] = []

        # Build UI
        self._create_left_panel()
        self._create_right_panel()

        # Start log queue processor
        self._process_log_queue()

        # Welcome message
        self._log_message("=" * 60)
        self._log_message("Welcome to Flood Hazard EAD Calculator v2.0!")
        self._log_message("=" * 60)
        self._log_message("")
        self._log_message("Steps:")
        self._log_message("1. Select the input Parquet file")
        self._log_message("2. Click [Analyze Input File] to read columns")
        self._log_message("3. Select water depth columns for each "
                          "return period in the popup dialog")
        self._log_message("4. Set output path and click [Start Processing]")
        self._log_message("")
        self._log_message("Formulas:")
        self._log_message("-" * 40)
        self._log_message("EAD = integral_0^1 H(p) dp  "
                          "(Trapezoidal integration)")
        self._log_message("Risk = (Hazard x Exposure x "
                          "Vulnerability)^(1/3)  (Geometric mean)")
        self._log_message("-" * 40)
        self._log_message("")

    def _create_left_panel(self):
        """Create the left control panel."""
        left_frame = CTkFrame(self, corner_radius=10)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_columnconfigure(0, weight=1)

        # Title
        CTkLabel(
            left_frame, text="Control Panel",
            font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        # Separator
        CTkFrame(left_frame, height=2, fg_color=COLORS['primary']
                 ).grid(row=1, column=0, padx=20, pady=5, sticky="ew")

        # === Input File Section ===
        input_section = CTkFrame(left_frame, fg_color="transparent")
        input_section.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        input_section.grid_columnconfigure(0, weight=1)

        CTkLabel(
            input_section, text="Input File (.parquet)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        input_entry_frame = CTkFrame(input_section, fg_color="transparent")
        input_entry_frame.grid(row=1, column=0, sticky="ew")
        input_entry_frame.grid_columnconfigure(0, weight=1)

        self.input_entry = CTkEntry(
            input_entry_frame, textvariable=self.input_path,
            height=35, font=ctk.CTkFont(size=12))
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        CTkButton(
            input_entry_frame, text="Browse", width=80, height=35,
            command=self._browse_input_file
        ).grid(row=0, column=1)

        # === Analyze Input File Button ===
        analyze_frame = CTkFrame(input_section, fg_color="transparent")
        analyze_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        analyze_frame.grid_columnconfigure(0, weight=1)

        self.analyze_btn = CTkButton(
            analyze_frame,
            text="Analyze Input File (Read Columns & Select Depth Fields)",
            height=42,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS['info'], hover_color="#0097A7",
            command=self._analyze_input_file)
        self.analyze_btn.grid(row=0, column=0, sticky="ew")

        # Mapping status label
        self.mapping_status_label = CTkLabel(
            input_section,
            text="No depth fields selected yet. "
                 "Please click 'Analyze Input File' first.",
            font=ctk.CTkFont(size=12),
            text_color=COLORS['warning'])
        self.mapping_status_label.grid(
            row=3, column=0, sticky="w", pady=(5, 0))

        # === Output Directory Section ===
        output_section = CTkFrame(left_frame, fg_color="transparent")
        output_section.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        output_section.grid_columnconfigure(0, weight=1)

        CTkLabel(
            output_section, text="Output Directory",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        output_entry_frame = CTkFrame(output_section, fg_color="transparent")
        output_entry_frame.grid(row=1, column=0, sticky="ew")
        output_entry_frame.grid_columnconfigure(0, weight=1)

        self.output_entry = CTkEntry(
            output_entry_frame, textvariable=self.output_dir,
            height=35, font=ctk.CTkFont(size=12))
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        CTkButton(
            output_entry_frame, text="Browse", width=80, height=35,
            command=self._browse_output_dir
        ).grid(row=0, column=1)

        # === Output Filename Section ===
        filename_section = CTkFrame(left_frame, fg_color="transparent")
        filename_section.grid(row=4, column=0, padx=20, pady=10,
                              sticky="ew")
        filename_section.grid_columnconfigure(0, weight=1)

        CTkLabel(
            filename_section, text="Output Filename",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.filename_entry = CTkEntry(
            filename_section, textvariable=self.output_file,
            height=35, font=ctk.CTkFont(size=12))
        self.filename_entry.grid(row=1, column=0, sticky="ew")

        # Separator
        CTkFrame(left_frame, height=2, fg_color=COLORS['secondary']
                 ).grid(row=5, column=0, padx=20, pady=15, sticky="ew")

        # === Processing Options ===
        options_section = CTkFrame(left_frame, fg_color="transparent")
        options_section.grid(row=6, column=0, padx=20, pady=10,
                             sticky="ew")

        CTkLabel(
            options_section, text="Processing Options",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        chunk_frame = CTkFrame(options_section, fg_color="transparent")
        chunk_frame.grid(row=1, column=0, sticky="ew", pady=5)

        CTkLabel(chunk_frame, text="Chunk Size:").grid(
            row=0, column=0, sticky="w")

        self.chunk_size_var = tk.StringVar(value="500000")
        CTkOptionMenu(
            chunk_frame, variable=self.chunk_size_var,
            values=["100000", "250000", "500000", "1000000"], width=120
        ).grid(row=0, column=1, padx=10)
        CTkLabel(chunk_frame, text="rows").grid(row=0, column=2)

        # Separator
        CTkFrame(left_frame, height=2, fg_color=COLORS['secondary']
                 ).grid(row=7, column=0, padx=20, pady=15, sticky="ew")

        # === Control Buttons ===
        button_frame = CTkFrame(left_frame, fg_color="transparent")
        button_frame.grid(row=8, column=0, padx=20, pady=10, sticky="ew")
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

        self.start_btn = CTkButton(
            button_frame, text="Start Processing", height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=COLORS['success'], hover_color="#388E3C",
            command=self._start_processing)
        self.start_btn.grid(row=0, column=0, padx=5, sticky="ew")

        self.stop_btn = CTkButton(
            button_frame, text="Stop", height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=COLORS['danger'], hover_color="#C62828",
            state="disabled", command=self._stop_processing)
        self.stop_btn.grid(row=0, column=1, padx=5, sticky="ew")

        # Validate button
        CTkButton(
            button_frame, text="Validate Input", height=40,
            font=ctk.CTkFont(size=14),
            fg_color=COLORS['info'], hover_color="#0097A7",
            command=self._validate_input
        ).grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

        # === Status Section ===
        status_section = CTkFrame(left_frame, fg_color="transparent")
        status_section.grid(row=9, column=0, padx=20, pady=10, sticky="ew")

        CTkLabel(
            status_section, text="Status",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.status_label = CTkLabel(
            status_section, text="Ready",
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_muted'])
        self.status_label.grid(row=1, column=0, sticky="w")

        # Progress bar
        self.progress_bar = CTkProgressBar(
            status_section, height=20,
            progress_color=COLORS['primary'])
        self.progress_bar.grid(row=2, column=0, sticky="ew", pady=(10, 5))
        self.progress_bar.set(0)

        self.progress_label = CTkLabel(
            status_section, text="0%",
            font=ctk.CTkFont(size=12))
        self.progress_label.grid(row=3, column=0, sticky="w")

        # === Formula Reference ===
        formula_section = CTkFrame(left_frame, corner_radius=8)
        formula_section.grid(row=10, column=0, padx=20, pady=20,
                             sticky="ew")

        CTkLabel(
            formula_section, text="Calculation Formulas",
            font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        formulas_text = (
            "EAD = integral H(p) dp\n"
            "    = Sum[(Hj + Hj+1)/2 * dp]\n\n"
            "Risk = (H x E x V)^(1/3)\n\n"
            "Return Periods: user-defined via field mapping"
        )

        CTkLabel(
            formula_section, text=formulas_text,
            font=ctk.CTkFont(size=11), justify="left",
            text_color=COLORS['text_muted']
        ).grid(row=1, column=0, padx=10, pady=(0, 10), sticky="w")

    def _create_right_panel(self):
        """Create the right log panel."""
        right_frame = CTkFrame(self, corner_radius=10)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        CTkLabel(
            right_frame, text="Processing Log",
            font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        self.log_textbox = CTkTextbox(
            right_frame,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word", activate_scrollbars=True)
        self.log_textbox.grid(row=1, column=0, padx=20, pady=(10, 10),
                              sticky="nsew")

        CTkButton(
            right_frame, text="Clear Log", height=35, width=120,
            command=self._clear_log
        ).grid(row=2, column=0, padx=20, pady=(0, 20), sticky="e")

    # ------------------------------------------------------------------
    # Analyze Input File -> Open field mapping dialog
    # ------------------------------------------------------------------
    def _analyze_input_file(self):
        """Read the parquet schema and open the field mapping dialog."""
        input_path = self.input_path.get().strip()

        if not input_path:
            messagebox.showerror("Error", "Please select an input file first!")
            return

        if not os.path.isfile(input_path):
            messagebox.showerror("Error",
                                 f"File does not exist:\n{input_path}")
            return

        self._log_message("\n" + "=" * 50)
        self._log_message("[ANALYZE] Reading input file schema...")

        try:
            parquet_file = pq.ParquetFile(input_path)
            schema = parquet_file.schema_arrow
            metadata = parquet_file.metadata
            column_names = [field.name for field in schema]

            self._file_columns = column_names

            self._log_message(
                f"[ANALYZE] File has {len(column_names)} columns, "
                f"{metadata.num_rows:,} rows")
            self._log_message("[ANALYZE] Column list:")
            for i, col in enumerate(column_names):
                dtype = schema.field(col).type
                self._log_message(f"       {i + 1:3d}. {col}  ({dtype})")

        except Exception as e:
            messagebox.showerror("Read Error",
                                 f"Cannot read file:\n{str(e)}")
            self._log_message(f"[ERROR] Read failed: {str(e)}")
            return

        # Pre-populate with current mappings if available
        current_mappings = (self.field_mappings
                            if self.field_mappings else None)

        # Open mapping dialog
        dialog = FieldMappingDialog(self, column_names, current_mappings)
        self.wait_window(dialog)

        if dialog.result is not None:
            self.field_mappings = dialog.result
            self.exposure_col = dialog.exposure_result
            self.vulnerability_col = dialog.vulnerability_result

            self._log_message("\n[ANALYZE] Field mapping confirmed:")
            for rp, col in self.field_mappings:
                self._log_message(
                    f"       RP {rp:.0f} yr  ->  {col}")
            self._log_message(
                f"       Exposure      ->  {self.exposure_col}")
            self._log_message(
                f"       Vulnerability ->  {self.vulnerability_col}")

            n = len(self.field_mappings)
            self.mapping_status_label.configure(
                text=(f"{n} depth field(s) selected. "
                      f"Click 'Analyze Input File' to modify."),
                text_color=COLORS['success'])
        else:
            self._log_message("[ANALYZE] User cancelled field mapping.")

    # ------------------------------------------------------------------
    # File browsing and validation
    # ------------------------------------------------------------------
    def _browse_input_file(self):
        """Open file dialog to select input file."""
        filename = filedialog.askopenfilename(
            title="Select Input Parquet File",
            filetypes=[("Parquet files", "*.parquet"),
                       ("All files", "*.*")],
            initialdir=os.path.dirname(self.input_path.get()))
        if filename:
            self.input_path.set(filename)
            self._log_message(f"[INPUT] Selected: {filename}")
            # Reset mapping when file changes
            self.field_mappings = []
            self._file_columns = []
            self.mapping_status_label.configure(
                text="File changed. Please click 'Analyze Input File' again.",
                text_color=COLORS['warning'])

    def _browse_output_dir(self):
        """Open dialog to select output directory."""
        dirname = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir.get())
        if dirname:
            self.output_dir.set(dirname)
            self._log_message(f"[OUTPUT] Directory: {dirname}")

    def _validate_input(self):
        """Validate the input file."""
        input_path = self.input_path.get()

        if not input_path:
            messagebox.showerror("Error",
                                 "Please select an input file.")
            return

        if not self.field_mappings:
            messagebox.showerror(
                "Error",
                "Please click 'Analyze Input File' first to "
                "select water depth columns!")
            return

        self._log_message("\n" + "=" * 50)
        self._log_message("[VALIDATE] Starting input validation...")

        hazard_cols = [col for _, col in self.field_mappings]
        return_periods = [rp for rp, _ in self.field_mappings]
        output_path = os.path.join(self.output_dir.get(),
                                   self.output_file.get())

        processor = DataProcessor(
            input_path=input_path,
            output_path=output_path,
            hazard_columns=hazard_cols,
            return_periods=return_periods,
            exposure_col=self.exposure_col,
            vulnerability_col=self.vulnerability_col,
            log_callback=self._log_message)

        is_valid, msg, meta = processor.validate_input()

        if is_valid:
            self._log_message(f"\n[SUCCESS] {msg}")
            self._log_message("[INFO] File is ready for processing.")
            self.status_label.configure(
                text="Validated", text_color=COLORS['success'])
        else:
            self._log_message(f"\n[ERROR] {msg}")
            self.status_label.configure(
                text="Validation Failed", text_color=COLORS['danger'])
            messagebox.showerror("Validation Error", msg)

    def _start_processing(self):
        """Start data processing in a background thread."""
        input_path = self.input_path.get()
        output_dir = self.output_dir.get()
        output_file = self.output_file.get()

        if not input_path:
            messagebox.showerror("Error",
                                 "Please select an input file.")
            return
        if not output_dir:
            messagebox.showerror("Error",
                                 "Please select an output directory.")
            return
        if not output_file:
            messagebox.showerror("Error",
                                 "Please specify an output filename.")
            return
        if not self.field_mappings:
            messagebox.showerror(
                "Error",
                "Please click 'Analyze Input File' first to "
                "select water depth columns!")
            return

        output_path = os.path.join(output_dir, output_file)
        chunk_size = int(self.chunk_size_var.get())

        hazard_cols = [col for _, col in self.field_mappings]
        return_periods = [rp for rp, _ in self.field_mappings]

        # Create processor with user-selected fields
        self.processor = DataProcessor(
            input_path=input_path,
            output_path=output_path,
            hazard_columns=hazard_cols,
            return_periods=return_periods,
            exposure_col=self.exposure_col,
            vulnerability_col=self.vulnerability_col,
            chunk_size=chunk_size,
            log_callback=self._log_message,
            progress_callback=self._update_progress)

        # Update UI state
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(
            text="Processing...", text_color=COLORS['warning'])
        self.progress_bar.set(0)

        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_worker, daemon=True)
        self.processing_thread.start()

    def _processing_worker(self):
        """Background worker for processing."""
        try:
            success, message = self.processor.process()
            self.after(0, lambda: self._processing_complete(
                success, message))
        except Exception as e:
            self.after(0, lambda: self._processing_complete(
                False, str(e)))

    def _processing_complete(self, success: bool, message: str):
        """Handle processing completion."""
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

        if success:
            self.status_label.configure(
                text="Complete", text_color=COLORS['success'])
            self.progress_bar.set(1)
            self.progress_label.configure(text="100%")
            messagebox.showinfo("Success", message)
        else:
            self.status_label.configure(
                text="Failed", text_color=COLORS['danger'])
            if "stopped" not in message.lower():
                messagebox.showerror("Error", message)

    def _stop_processing(self):
        """Stop processing."""
        if self.processor:
            self.processor.stop()
            self._log_message("[USER] Stop requested...")

    def _log_message(self, message: str):
        """Add message to log queue (thread-safe)."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}")

    def _process_log_queue(self):
        """Process log queue and update textbox."""
        while not self.log_queue.empty():
            try:
                message = self.log_queue.get_nowait()
                self.log_textbox.insert("end", message + "\n")
                self.log_textbox.see("end")
            except queue.Empty:
                break
        self.after(100, self._process_log_queue)

    def _update_progress(self, progress: float, message: str):
        """Update progress bar (thread-safe)."""
        self.after(0, lambda: self._set_progress(progress, message))

    def _set_progress(self, progress: float, message: str):
        """Set progress bar value."""
        self.progress_bar.set(progress)
        self.progress_label.configure(text=f"{progress * 100:.1f}%")
        self.status_label.configure(text=message)

    def _clear_log(self):
        """Clear the log textbox."""
        self.log_textbox.delete("1.0", "end")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Main entry point for the application."""
    print("=" * 60)
    print("Flood Hazard EAD Calculator v2.0")
    print("=" * 60)
    print("Starting application...")

    app = FloodEADCalculatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
