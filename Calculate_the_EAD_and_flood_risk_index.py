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
- EAD calculation using trapezoidal integration
- Composite Risk Index calculation
- Progress monitoring and detailed logging
- Statistical analysis and visualization export

Author: LONG
Date: 2026
Version: 1.0.0

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

# Configure warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================

# Application Configuration
APP_TITLE = "Flood Hazard EAD Calculator v1.0"
APP_WIDTH = 1400
APP_HEIGHT = 900
THEME_MODE = "dark"
COLOR_THEME = "blue"

# Processing Configuration
CHUNK_SIZE = 500_000  # Rows per chunk for memory efficiency
MAX_MEMORY_PERCENT = 75  # Maximum memory usage percentage
NUM_WORKERS = 8  # Number of parallel workers (adjust based on CPU)

# Return Periods Configuration
RETURN_PERIODS = [10, 20, 50, 75, 100, 200, 500]
HAZARD_COLUMNS = [
    'Hazard_Index_RP010',
    'Hazard_Index_RP020',
    'Hazard_Index_RP050',
    'Hazard_Index_RP075',
    'Hazard_Index_RP100',
    'Hazard_Index_RP200',
    'Hazard_Index_RP500'
]

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

    def __init__(self, return_periods: List[float] = RETURN_PERIODS):
        """
        Initialize the EAD Calculator.

        Args:
            return_periods: List of return periods in years [10, 20, 50, 75, 100, 200, 500]
        """
        self.return_periods = np.array(return_periods, dtype=np.float64)
        self.exceedance_probs = 1.0 / self.return_periods

        # Pre-compute boundary-extended probabilities
        # Sort by probability (ascending)
        sort_idx = np.argsort(self.exceedance_probs)
        self.sorted_probs = self.exceedance_probs[sort_idx]
        self.sort_indices = sort_idx

        # Extended probability array with boundaries [0, p1, p2, ..., pn, 1]
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

        This is the optimized method for processing large datasets.
        Uses NumPy broadcasting for efficient computation.

        Args:
            hazard_matrix: 2D array of shape (n_cells, n_return_periods)
                           Each row contains hazard values for one grid cell
                           Columns correspond to return periods

        Returns:
            1D array of normalized EAD values (0-1 range)
        """
        n_cells = hazard_matrix.shape[0]

        # Sort by probability (same order for all cells)
        H_sorted = hazard_matrix[:, self.sort_indices]

        # Add boundary conditions
        # Low-frequency (p=0): use value at 500-year RP
        # High-frequency (p=1): use value at 10-year RP
        H_extended = np.column_stack([
            H_sorted[:, 0],  # Boundary at p=0 (extend from 500-yr)
            H_sorted,  # Original sorted values
            H_sorted[:, -1]  # Boundary at p=1 (extend from 10-yr)
        ])

        # Trapezoidal integration: area = (H_i + H_{i+1})/2 * delta_p
        avg_H = (H_extended[:, :-1] + H_extended[:, 1:]) / 2
        ead_values = np.sum(avg_H * self.delta_p, axis=1)

        # Normalize by EAD_max
        normalized_ead = ead_values / self.ead_max

        # Clip to [0, 1] range
        normalized_ead = np.clip(normalized_ead, 0.0, 1.0)

        return normalized_ead.astype(np.float32)

    def calculate_risk_index(self,
                             hazard: np.ndarray,
                             exposure: np.ndarray,
                             vulnerability: np.ndarray) -> np.ndarray:
        """
        Calculate Composite Flood Risk Index.

        Formula: Risk_i = (Hazard_i × Exposure_i × Vulnerability_i)^(1/3)

        This is the geometric mean formulation from the IPCC framework.

        Args:
            hazard: Normalized hazard index (Hazard_Index_EAD)
            exposure: Exposure index (Exposure_Index)
            vulnerability: Vulnerability index (Vulnerability_Index)

        Returns:
            Composite Risk Index array
        """
        # Handle NaN and negative values
        hazard = np.nan_to_num(hazard, nan=0.0)
        exposure = np.nan_to_num(exposure, nan=0.0)
        vulnerability = np.nan_to_num(vulnerability, nan=0.0)

        # Clip to valid range
        hazard = np.clip(hazard, 0.0, 1.0)
        exposure = np.clip(exposure, 0.0, 1.0)
        vulnerability = np.clip(vulnerability, 0.0, 1.0)

        # Calculate geometric mean (cube root of product)
        product = hazard * exposure * vulnerability
        risk_index = np.power(product, 1 / 3)

        return risk_index.astype(np.float32)


# ==============================================================================
# DATA PROCESSOR (Memory-Efficient Chunked Processing)
# ==============================================================================

class DataProcessor:
    """
    Memory-efficient processor for large Parquet datasets.

    Processes data in chunks to prevent memory overflow on datasets
    exceeding 30GB. Optimized for systems with 64GB RAM.
    """

    def __init__(self,
                 input_path: str,
                 output_path: str,
                 chunk_size: int = CHUNK_SIZE,
                 log_callback=None,
                 progress_callback=None):
        """
        Initialize the Data Processor.

        Args:
            input_path: Path to input Parquet file
            output_path: Path to output Parquet file
            chunk_size: Number of rows per processing chunk
            log_callback: Function for logging messages
            progress_callback: Function for progress updates
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.chunk_size = chunk_size
        self.log = log_callback or print
        self.progress = progress_callback or (lambda x, y: None)

        self.ead_calculator = EADCalculator()
        self._stop_flag = False
        self._output_schema = None  # Store consistent schema

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
            mem_info = process.memory_info()
            mem_gb = mem_info.rss / (1024 ** 3)
            self.log(f"[MEMORY] Current usage: {mem_gb:.2f} GB")
        except ImportError:
            pass

    def _create_output_schema(self, input_schema):
        """
        Create a consistent output schema based on input schema.

        Converts integer columns that may have NaN to float to ensure
        consistency across all chunks.

        Args:
            input_schema: PyArrow schema from input file

        Returns:
            PyArrow schema for output file
        """
        # Columns that may have NaN and need to be float
        nullable_int_columns = {
            'Income_Classification', 'Country', 'Sovereig', 'Developed'
        }

        new_fields = []
        for field in input_schema:
            if field.name in nullable_int_columns:
                # Convert to float64 to handle NaN consistently
                new_fields.append(pa.field(field.name, pa.float64()))
            else:
                new_fields.append(field)

        # Add new calculated columns
        new_fields.append(pa.field('Hazard_Index_EAD', pa.float32()))
        new_fields.append(pa.field('Risk_Index', pa.float32()))

        return pa.schema(new_fields)

    def _ensure_schema_consistency(self, df, schema):
        """
        Ensure DataFrame matches the expected schema.

        Args:
            df: pandas DataFrame
            schema: Target PyArrow schema

        Returns:
            DataFrame with consistent types
        """
        for field in schema:
            col_name = field.name
            if col_name not in df.columns:
                continue

            # Convert based on PyArrow type
            pa_type = field.type

            if pa.types.is_float64(pa_type):
                df[col_name] = df[col_name].astype('float64')
            elif pa.types.is_float32(pa_type):
                df[col_name] = df[col_name].astype('float32')
            elif pa.types.is_int64(pa_type):
                # For int columns, fill NaN with -1 or convert to float
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
        """
        Validate the input Parquet file.

        Returns:
            Tuple of (is_valid, message, metadata)
        """
        self.log("[VALIDATION] Starting input file validation...")

        if not self.input_path.exists():
            return False, f"Input file not found: {self.input_path}", {}

        try:
            # Read metadata without loading data
            parquet_file = pq.ParquetFile(self.input_path)
            metadata = parquet_file.metadata
            schema = parquet_file.schema_arrow

            # Get basic info
            num_rows = metadata.num_rows
            num_columns = metadata.num_columns
            file_size_gb = self.input_path.stat().st_size / (1024 ** 3)

            self.log(f"[VALIDATION] File size: {file_size_gb:.2f} GB")
            self.log(f"[VALIDATION] Total rows: {num_rows:,}")
            self.log(f"[VALIDATION] Total columns: {num_columns}")

            # Check for required columns
            column_names = [field.name for field in schema]

            required_columns = (
                    HAZARD_COLUMNS +
                    ['Exposure_Index', 'Vulnerability_Index']
            )

            missing_columns = []
            for col in required_columns:
                if col not in column_names:
                    missing_columns.append(col)

            if missing_columns:
                return False, f"Missing required columns: {missing_columns}", {}

            self.log("[VALIDATION] All required columns found!")

            # Calculate estimated chunks
            num_chunks = (num_rows + self.chunk_size - 1) // self.chunk_size
            self.log(f"[VALIDATION] Estimated processing chunks: {num_chunks}")

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
        """
        Process the entire Parquet file in chunks.

        Returns:
            Tuple of (success, message)
        """
        start_time = time.time()
        self.log("=" * 70)
        self.log("[PROCESS] Starting EAD and Risk Index calculation...")
        self.log("=" * 70)

        # Validate input
        is_valid, msg, meta = self.validate_input()
        if not is_valid:
            return False, msg

        num_rows = meta['num_rows']
        num_chunks = meta['num_chunks']

        try:
            # Open input file for chunked reading
            parquet_file = pq.ParquetFile(self.input_path)
            input_schema = parquet_file.schema_arrow

            # Create consistent output schema
            self.log("[PROCESS] Creating consistent output schema...")
            self._output_schema = self._create_output_schema(input_schema)
            self.log(f"[PROCESS] Output schema has {len(self._output_schema)} fields")

            # Create output directory if needed
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Temporary file for writing
            temp_output = self.output_path.with_suffix('.tmp.parquet')

            # Initialize writer (will be created on first chunk)
            writer = None
            processed_rows = 0
            chunk_idx = 0

            self.log(f"[PROCESS] Starting chunked processing...")
            self.log(f"[PROCESS] Chunk size: {self.chunk_size:,} rows")

            # Process in batches
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                if self._stop_flag:
                    self.log("[PROCESS] Stopping processing by user request...")
                    if writer:
                        writer.close()
                    if temp_output.exists():
                        temp_output.unlink()
                    return False, "Processing stopped by user"

                chunk_start_time = time.time()
                chunk_idx += 1

                # Convert batch to pandas DataFrame
                df = batch.to_pandas()
                chunk_rows = len(df)

                self.log(f"\n[CHUNK {chunk_idx}/{num_chunks}] Processing {chunk_rows:,} rows...")

                # Extract hazard columns
                hazard_matrix = df[HAZARD_COLUMNS].values.astype(np.float64)

                # Handle NaN values
                hazard_matrix = np.nan_to_num(hazard_matrix, nan=0.0)

                # Calculate EAD
                self.log(f"[CHUNK {chunk_idx}] Calculating EAD using trapezoidal integration...")
                ead_values = self.ead_calculator.calculate_ead_vectorized(hazard_matrix)

                # Add Hazard_Index_EAD column
                df['Hazard_Index_EAD'] = ead_values.astype(np.float32)

                # Calculate Risk Index
                self.log(f"[CHUNK {chunk_idx}] Calculating composite Risk Index...")
                exposure = df['Exposure_Index'].values.astype(np.float64)
                vulnerability = df['Vulnerability_Index'].values.astype(np.float64)

                risk_values = self.ead_calculator.calculate_risk_index(
                    ead_values, exposure, vulnerability
                )

                # Add Risk_Index column
                df['Risk_Index'] = risk_values.astype(np.float32)

                # Ensure schema consistency before writing
                self.log(f"[CHUNK {chunk_idx}] Ensuring schema consistency...")
                df = self._ensure_schema_consistency(df, self._output_schema)

                # Convert to Arrow Table with explicit schema
                # Only include columns that are in the schema
                schema_columns = [field.name for field in self._output_schema]
                df_filtered = df[[col for col in schema_columns if col in df.columns]]

                table = pa.Table.from_pandas(df_filtered, schema=self._output_schema, preserve_index=False)

                # Initialize or append to writer
                if writer is None:
                    writer = pq.ParquetWriter(
                        str(temp_output),
                        self._output_schema,
                        compression='snappy'
                    )

                writer.write_table(table)

                # Update progress
                processed_rows += chunk_rows
                progress_pct = processed_rows / num_rows
                self.progress(progress_pct, f"Processed {processed_rows:,}/{num_rows:,} rows")

                # Log timing
                chunk_time = time.time() - chunk_start_time
                rows_per_sec = chunk_rows / chunk_time
                self.log(f"[CHUNK {chunk_idx}] Completed in {chunk_time:.2f}s ({rows_per_sec:,.0f} rows/sec)")

                # Memory cleanup
                del df, df_filtered, hazard_matrix, ead_values, risk_values, table
                gc.collect()

                # Log memory usage every 10 chunks
                if chunk_idx % 10 == 0:
                    self._log_memory_usage()

            # Close writer
            if writer:
                writer.close()

            # Rename temp file to final output
            if temp_output.exists():
                if self.output_path.exists():
                    self.output_path.unlink()
                temp_output.rename(self.output_path)

            # Final summary
            total_time = time.time() - start_time
            self.log("\n" + "=" * 70)
            self.log("[COMPLETE] Processing finished successfully!")
            self.log(f"[COMPLETE] Total rows processed: {processed_rows:,}")
            self.log(f"[COMPLETE] Total time: {total_time:.2f} seconds")
            self.log(f"[COMPLETE] Average speed: {processed_rows / total_time:,.0f} rows/sec")
            self.log(f"[COMPLETE] Output saved to: {self.output_path}")
            self.log("=" * 70)

            return True, "Processing completed successfully!"

        except Exception as e:
            self.log(f"[ERROR] Processing failed: {str(e)}")
            import traceback
            self.log(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            return False, f"Processing error: {str(e)}"


# ==============================================================================
# GUI APPLICATION
# ==============================================================================

class FloodEADCalculatorApp(CTk):
    """
    Main GUI Application for Flood EAD Calculator.

    Features a modern dark theme interface with:
    - File selection panel
    - Processing controls
    - Real-time logging
    - Progress tracking
    """

    def __init__(self):
        super().__init__()

        # Configure window
        self.title(APP_TITLE)
        self.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.minsize(1200, 700)

        # Set appearance
        ctk.set_appearance_mode(THEME_MODE)
        ctk.set_default_color_theme(COLOR_THEME)

        # Configure grid weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)

        # Initialize variables
        self.input_path = tk.StringVar(value=DEFAULT_INPUT_PATH)
        self.output_dir = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        self.output_file = tk.StringVar(value=DEFAULT_OUTPUT_FILE)
        self.processor: Optional[DataProcessor] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.log_queue = queue.Queue()

        # Build UI
        self._create_left_panel()
        self._create_right_panel()

        # Start log queue processor
        self._process_log_queue()

        # Welcome message
        self._log_message("=" * 60)
        self._log_message("Welcome to Flood Hazard EAD Calculator!")
        self._log_message("=" * 60)
        self._log_message("")
        self._log_message("This application calculates:")
        self._log_message("1. Expected Annual Damage (EAD) - Hazard_Index_EAD")
        self._log_message("2. Composite Risk Index - Risk_Index")
        self._log_message("")
        self._log_message("Mathematical Formulas:")
        self._log_message("-" * 40)
        self._log_message("EAD = integral_0^1 H(p) dp")
        self._log_message("    (Trapezoidal integration)")
        self._log_message("")
        self._log_message("Risk = (Hazard × Exposure × Vulnerability)^(1/3)")
        self._log_message("    (Geometric mean)")
        self._log_message("-" * 40)
        self._log_message("")
        self._log_message("Please select input file and output location,")
        self._log_message("then click [Start Processing] to begin.")
        self._log_message("")

    def _create_left_panel(self):
        """Create the left control panel."""
        # Main frame for left panel
        left_frame = CTkFrame(self, corner_radius=10)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_columnconfigure(0, weight=1)

        # Title
        title_label = CTkLabel(
            left_frame,
            text="📊 Control Panel",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        # Separator
        sep1 = CTkFrame(left_frame, height=2, fg_color=COLORS['primary'])
        sep1.grid(row=1, column=0, padx=20, pady=5, sticky="ew")

        # === Input File Section ===
        input_section = CTkFrame(left_frame, fg_color="transparent")
        input_section.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        input_section.grid_columnconfigure(0, weight=1)

        CTkLabel(
            input_section,
            text="📁 Input File (.parquet)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        input_entry_frame = CTkFrame(input_section, fg_color="transparent")
        input_entry_frame.grid(row=1, column=0, sticky="ew")
        input_entry_frame.grid_columnconfigure(0, weight=1)

        self.input_entry = CTkEntry(
            input_entry_frame,
            textvariable=self.input_path,
            height=35,
            font=ctk.CTkFont(size=12)
        )
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        browse_input_btn = CTkButton(
            input_entry_frame,
            text="Browse",
            width=80,
            height=35,
            command=self._browse_input_file
        )
        browse_input_btn.grid(row=0, column=1)

        # === Output Directory Section ===
        output_section = CTkFrame(left_frame, fg_color="transparent")
        output_section.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        output_section.grid_columnconfigure(0, weight=1)

        CTkLabel(
            output_section,
            text="📂 Output Directory",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        output_entry_frame = CTkFrame(output_section, fg_color="transparent")
        output_entry_frame.grid(row=1, column=0, sticky="ew")
        output_entry_frame.grid_columnconfigure(0, weight=1)

        self.output_entry = CTkEntry(
            output_entry_frame,
            textvariable=self.output_dir,
            height=35,
            font=ctk.CTkFont(size=12)
        )
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        browse_output_btn = CTkButton(
            output_entry_frame,
            text="Browse",
            width=80,
            height=35,
            command=self._browse_output_dir
        )
        browse_output_btn.grid(row=0, column=1)

        # === Output Filename Section ===
        filename_section = CTkFrame(left_frame, fg_color="transparent")
        filename_section.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        filename_section.grid_columnconfigure(0, weight=1)

        CTkLabel(
            filename_section,
            text="📝 Output Filename",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.filename_entry = CTkEntry(
            filename_section,
            textvariable=self.output_file,
            height=35,
            font=ctk.CTkFont(size=12)
        )
        self.filename_entry.grid(row=1, column=0, sticky="ew")

        # Separator
        sep2 = CTkFrame(left_frame, height=2, fg_color=COLORS['secondary'])
        sep2.grid(row=5, column=0, padx=20, pady=15, sticky="ew")

        # === Processing Options ===
        options_section = CTkFrame(left_frame, fg_color="transparent")
        options_section.grid(row=6, column=0, padx=20, pady=10, sticky="ew")

        CTkLabel(
            options_section,
            text="⚙️ Processing Options",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        # Chunk size option
        chunk_frame = CTkFrame(options_section, fg_color="transparent")
        chunk_frame.grid(row=1, column=0, sticky="ew", pady=5)

        CTkLabel(chunk_frame, text="Chunk Size:").grid(row=0, column=0, sticky="w")

        self.chunk_size_var = tk.StringVar(value="500000")
        chunk_menu = CTkOptionMenu(
            chunk_frame,
            variable=self.chunk_size_var,
            values=["100000", "250000", "500000", "1000000"],
            width=120
        )
        chunk_menu.grid(row=0, column=1, padx=10)
        CTkLabel(chunk_frame, text="rows").grid(row=0, column=2)

        # Separator
        sep3 = CTkFrame(left_frame, height=2, fg_color=COLORS['secondary'])
        sep3.grid(row=7, column=0, padx=20, pady=15, sticky="ew")

        # === Control Buttons ===
        button_frame = CTkFrame(left_frame, fg_color="transparent")
        button_frame.grid(row=8, column=0, padx=20, pady=10, sticky="ew")
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

        self.start_btn = CTkButton(
            button_frame,
            text="▶ Start Processing",
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=COLORS['success'],
            hover_color="#388E3C",
            command=self._start_processing
        )
        self.start_btn.grid(row=0, column=0, padx=5, sticky="ew")

        self.stop_btn = CTkButton(
            button_frame,
            text="⏹ Stop",
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=COLORS['danger'],
            hover_color="#C62828",
            state="disabled",
            command=self._stop_processing
        )
        self.stop_btn.grid(row=0, column=1, padx=5, sticky="ew")

        # Validate button
        validate_btn = CTkButton(
            button_frame,
            text="🔍 Validate Input",
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color=COLORS['info'],
            hover_color="#0097A7",
            command=self._validate_input
        )
        validate_btn.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

        # === Status Section ===
        status_section = CTkFrame(left_frame, fg_color="transparent")
        status_section.grid(row=9, column=0, padx=20, pady=10, sticky="ew")

        CTkLabel(
            status_section,
            text="📈 Status",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.status_label = CTkLabel(
            status_section,
            text="Ready",
            font=ctk.CTkFont(size=12),
            text_color=COLORS['text_muted']
        )
        self.status_label.grid(row=1, column=0, sticky="w")

        # Progress bar
        self.progress_bar = CTkProgressBar(
            status_section,
            height=20,
            progress_color=COLORS['primary']
        )
        self.progress_bar.grid(row=2, column=0, sticky="ew", pady=(10, 5))
        self.progress_bar.set(0)

        self.progress_label = CTkLabel(
            status_section,
            text="0%",
            font=ctk.CTkFont(size=12)
        )
        self.progress_label.grid(row=3, column=0, sticky="w")

        # === Formula Reference ===
        formula_section = CTkFrame(left_frame, corner_radius=8)
        formula_section.grid(row=10, column=0, padx=20, pady=20, sticky="ew")

        CTkLabel(
            formula_section,
            text="📐 Calculation Formulas",
            font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        formulas_text = """
EAD = ∫₀¹ H(p) dp ≈ Σ[(Hⱼ + Hⱼ₊₁)/2 × Δp]

Risk = (H × E × V)^(1/3)

Return Periods: 10, 20, 50, 75, 100, 200, 500 years
        """

        CTkLabel(
            formula_section,
            text=formulas_text,
            font=ctk.CTkFont(size=11),
            justify="left",
            text_color=COLORS['text_muted']
        ).grid(row=1, column=0, padx=10, pady=(0, 10), sticky="w")

    def _create_right_panel(self):
        """Create the right information panel."""
        # Main frame for right panel
        right_frame = CTkFrame(self, corner_radius=10)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Title
        title_label = CTkLabel(
            right_frame,
            text="📋 Processing Log",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        # Log textbox with scrollbar
        self.log_textbox = CTkTextbox(
            right_frame,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word",
            activate_scrollbars=True
        )
        self.log_textbox.grid(row=1, column=0, padx=20, pady=(10, 10), sticky="nsew")

        # Clear log button
        clear_btn = CTkButton(
            right_frame,
            text="🗑 Clear Log",
            height=35,
            width=120,
            command=self._clear_log
        )
        clear_btn.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="e")

    def _browse_input_file(self):
        """Open file dialog to select input file."""
        filename = filedialog.askopenfilename(
            title="Select Input Parquet File",
            filetypes=[("Parquet files", "*.parquet"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.input_path.get())
        )
        if filename:
            self.input_path.set(filename)
            self._log_message(f"[INPUT] Selected: {filename}")

    def _browse_output_dir(self):
        """Open dialog to select output directory."""
        dirname = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir.get()
        )
        if dirname:
            self.output_dir.set(dirname)
            self._log_message(f"[OUTPUT] Directory: {dirname}")

    def _validate_input(self):
        """Validate the input file."""
        input_path = self.input_path.get()

        if not input_path:
            messagebox.showerror("Error", "Please select an input file.")
            return

        self._log_message("\n" + "=" * 50)
        self._log_message("[VALIDATION] Starting input validation...")

        # Create temporary processor for validation
        output_path = os.path.join(self.output_dir.get(), self.output_file.get())
        processor = DataProcessor(
            input_path=input_path,
            output_path=output_path,
            log_callback=self._log_message
        )

        is_valid, msg, meta = processor.validate_input()

        if is_valid:
            self._log_message(f"\n[SUCCESS] {msg}")
            self._log_message(f"[INFO] File is ready for processing.")
            self.status_label.configure(text="Validated ✓", text_color=COLORS['success'])
        else:
            self._log_message(f"\n[ERROR] {msg}")
            self.status_label.configure(text="Validation Failed ✗", text_color=COLORS['danger'])
            messagebox.showerror("Validation Error", msg)

    def _start_processing(self):
        """Start the data processing in a background thread."""
        input_path = self.input_path.get()
        output_dir = self.output_dir.get()
        output_file = self.output_file.get()

        # Validate inputs
        if not input_path:
            messagebox.showerror("Error", "Please select an input file.")
            return

        if not output_dir:
            messagebox.showerror("Error", "Please select an output directory.")
            return

        if not output_file:
            messagebox.showerror("Error", "Please specify an output filename.")
            return

        output_path = os.path.join(output_dir, output_file)
        chunk_size = int(self.chunk_size_var.get())

        # Create processor
        self.processor = DataProcessor(
            input_path=input_path,
            output_path=output_path,
            chunk_size=chunk_size,
            log_callback=self._log_message,
            progress_callback=self._update_progress
        )

        # Update UI state
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="Processing...", text_color=COLORS['warning'])
        self.progress_bar.set(0)

        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_worker,
            daemon=True
        )
        self.processing_thread.start()

    def _processing_worker(self):
        """Background worker for processing."""
        try:
            success, message = self.processor.process()

            # Update UI on completion
            self.after(0, lambda: self._processing_complete(success, message))

        except Exception as e:
            self.after(0, lambda: self._processing_complete(False, str(e)))

    def _processing_complete(self, success: bool, message: str):
        """Handle processing completion."""
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

        if success:
            self.status_label.configure(text="Complete ✓", text_color=COLORS['success'])
            self.progress_bar.set(1)
            self.progress_label.configure(text="100%")
            messagebox.showinfo("Success", message)
        else:
            self.status_label.configure(text="Failed ✗", text_color=COLORS['danger'])
            if "stopped" not in message.lower():
                messagebox.showerror("Error", message)

    def _stop_processing(self):
        """Stop the processing."""
        if self.processor:
            self.processor.stop()
            self._log_message("[USER] Stop requested...")

    def _log_message(self, message: str):
        """Add message to log queue (thread-safe)."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_queue.put(formatted_message)

    def _process_log_queue(self):
        """Process log queue and update textbox."""
        while not self.log_queue.empty():
            try:
                message = self.log_queue.get_nowait()
                self.log_textbox.insert("end", message + "\n")
                self.log_textbox.see("end")
            except queue.Empty:
                break

        # Schedule next check
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
    print("Flood Hazard EAD Calculator")
    print("=" * 60)
    print("Starting application...")

    # Create and run application
    app = FloodEADCalculatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()