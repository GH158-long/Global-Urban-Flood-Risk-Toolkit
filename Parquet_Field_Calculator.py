import sys
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import traceback
import queue
import re
import copy

import polars as pl
import pyarrow.parquet as pq

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog

# ============================================================================
# Configuration
# ============================================================================

APP_NAME = "Parquet Field Calculator Pro"
APP_VERSION = "5.1.0"
DEFAULT_INPUT = r"E:\Global_Flood\Parquet\BUILT_AGE_V2.parquet"
DEFAULT_OUTPUT = r"E:\Global_Flood\FIG\DATA"

CHUNK_SIZE = 10_000_000
NUM_THREADS = 28

# ============================================================================
# Font Configuration - All Enlarged
# ============================================================================

FONT_TITLE = ('Segoe UI', 26, 'bold')
FONT_HEADING = ('Segoe UI', 18, 'bold')
FONT_SUBHEADING = ('Segoe UI', 16, 'bold')
FONT_BODY = ('Segoe UI', 14)
FONT_BODY_BOLD = ('Segoe UI', 14, 'bold')
FONT_SMALL = ('Segoe UI', 13)
FONT_SMALL_BOLD = ('Segoe UI', 13, 'bold')
FONT_TINY = ('Segoe UI', 12)
FONT_CODE = ('Consolas', 14)
FONT_CODE_SMALL = ('Consolas', 13)
FONT_BUTTON = ('Segoe UI', 13, 'bold')
FONT_BUTTON_LARGE = ('Segoe UI', 14, 'bold')
FONT_ICON = ('Segoe UI Emoji', 18)
FONT_ICON_LARGE = ('Segoe UI Emoji', 22)
FONT_ICON_XLARGE = ('Segoe UI Emoji', 28)
FONT_LIST = ('Consolas', 15)


# ============================================================================
# Colors
# ============================================================================

class Colors:
    BG_MAIN = '#F8FAFC'
    BG_CARD = '#FFFFFF'
    BG_INPUT = '#F8FAFC'
    BG_HOVER = '#F1F5F9'
    BG_SELECTED = '#DBEAFE'
    BG_DARK = '#0F172A'

    TEXT_PRIMARY = '#0F172A'
    TEXT_SECONDARY = '#475569'
    TEXT_MUTED = '#94A3B8'
    TEXT_WHITE = '#FFFFFF'
    TEXT_CODE = '#34D399'
    TEXT_HEADING = '#1E293B'

    BLUE = '#3B82F6'
    BLUE_LIGHT = '#EFF6FF'
    BLUE_DARK = '#1D4ED8'

    GREEN = '#10B981'
    GREEN_LIGHT = '#ECFDF5'
    GREEN_DARK = '#059669'

    RED = '#EF4444'
    RED_LIGHT = '#FEF2F2'

    ORANGE = '#F97316'
    ORANGE_LIGHT = '#FFF7ED'

    PURPLE = '#8B5CF6'
    PURPLE_LIGHT = '#F5F3FF'

    CYAN = '#06B6D4'
    CYAN_LIGHT = '#ECFEFF'

    INDIGO = '#6366F1'
    INDIGO_LIGHT = '#EEF2FF'

    TEAL = '#14B8A6'

    BORDER = '#E2E8F0'
    BORDER_FOCUS = '#3B82F6'

    BTN_OPERATOR = '#EFF6FF'
    BTN_NUMBER = '#F8FAFC'
    BTN_FUNCTION = '#FFF7ED'
    BTN_FIELD_NUM = '#ECFDF5'
    BTN_FIELD_STR = '#F5F3FF'


# ============================================================================
# Icons
# ============================================================================

class Icons:
    FILE = "📄"
    FOLDER = "📁"
    FOLDER_OPEN = "📂"
    SAVE = "💾"
    PLAY = "▶"
    STOP = "⏹"
    SEARCH = "🔍"
    ADD = "➕"
    DELETE = "🗑️"
    EDIT = "✏️"
    CHECK = "✓"
    CLOSE = "✕"
    TABLE = "📊"
    FIELD = "📋"
    NUMBER = "🔢"
    TEXT = "📝"
    CALC = "🧮"
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    MOVE_UP = "⬆"
    MOVE_DOWN = "⬇"
    SORT = "↕️"
    ROCKET = "🚀"
    SPARKLE = "✨"
    LIGHTNING = "⚡"
    CLOCK = "🕐"
    EYE = "👁"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FieldInfo:
    name: str
    dtype: str
    null_count: int
    sample_values: List[Any]
    is_new: bool = False
    is_numeric: bool = False


# ============================================================================
# Expression Engine
# ============================================================================

class ExpressionEngine:
    @staticmethod
    def validate(expr: str, fields: List[str]) -> Tuple[bool, str]:
        if not expr.strip():
            return False, "Expression is empty"
        if expr.count('(') != expr.count(')'):
            return False, "Unbalanced parentheses"
        if expr.count('[') != expr.count(']'):
            return False, "Unbalanced brackets"

        refs = re.findall(r'\[([^\]]+)\]', expr)
        for ref in refs:
            if ref not in fields:
                return False, f"Unknown field: [{ref}]"
        return True, "Valid expression"

    @staticmethod
    def to_polars(expr: str) -> str:
        result = re.sub(r'\[([^\]]+)\]', r'pl.col("\1")', expr)
        result = result.replace('^', '**')

        funcs = {
            r'\babs\(': 'ABS(', r'\bsqrt\(': 'SQRT(', r'\blog10\(': 'LOG10(',
            r'\blog\(': 'LOG(', r'\bln\(': 'LOG(', r'\bexp\(': 'EXP(',
            r'\bsin\(': 'SIN(', r'\bcos\(': 'COS(', r'\btan\(': 'TAN(',
            r'\bceil\(': 'CEIL(', r'\bfloor\(': 'FLOOR(', r'\bround\(': 'ROUND(',
            r'\bsign\(': 'SIGN(', r'\bmin\(': 'MIN(', r'\bmax\(': 'MAX(',
        }
        for p, r in funcs.items():
            result = re.sub(p, r, result, flags=re.IGNORECASE)
        return result

    @staticmethod
    def execute(df: pl.LazyFrame, expr: str, name: str) -> pl.LazyFrame:
        polars_str = ExpressionEngine.to_polars(expr)

        ctx = {
            'pl': pl,
            'ABS': lambda x: x.abs(), 'SQRT': lambda x: x.sqrt(),
            'LOG': lambda x: x.log(), 'LOG10': lambda x: x.log(10),
            'EXP': lambda x: x.exp(), 'SIN': lambda x: x.sin(),
            'COS': lambda x: x.cos(), 'TAN': lambda x: x.tan(),
            'CEIL': lambda x: x.ceil(), 'FLOOR': lambda x: x.floor(),
            'ROUND': lambda x: x.round(0), 'SIGN': lambda x: x.sign(),
            'MIN': lambda a, b: pl.min_horizontal([a, b]),
            'MAX': lambda a, b: pl.max_horizontal([a, b]),
        }

        result = eval(polars_str, ctx)
        return df.with_columns(result.alias(name))


# ============================================================================
# Worker Thread
# ============================================================================

class Worker(threading.Thread):
    def __init__(self, q: queue.Queue):
        super().__init__(daemon=True)
        self.q = q
        self.input_path = None
        self.output_path = None
        self.task = None
        self.expr = None
        self.result_name = None
        self.del_fields = []
        self.new_field = None
        self.rename_map = None
        self.new_order = None
        self.df = None
        self._stop = False
        self._ready = threading.Event()

    def stop(self):
        self._stop = True
        self._msg('log', f"{Icons.WARNING} Stopping...")

    def _ts(self):
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _msg(self, t, d):
        self.q.put((t, d))

    def _log(self, m):
        self._msg('log', f"[{self._ts()}] {m}")

    def start_task(self, t):
        self.task = t
        self._stop = False
        self._ready.set()

    def run(self):
        while True:
            self._ready.wait()
            self._ready.clear()
            if self._stop:
                continue
            try:
                task_map = {
                    'analyze': self._analyze,
                    'execute': self._execute,
                    'delete': self._delete,
                    'add': self._add,
                    'save': self._save,
                    'preview': self._preview,
                    'manage': self._manage_fields,
                }
                task_map.get(self.task, lambda: None)()
            except Exception as e:
                self._log(f"{Icons.ERROR} Error: {e}")
                self._log(traceback.format_exc())
                self._msg('done', (False, str(e)))

    def _analyze(self):
        self._log(f"{Icons.SEARCH} Analyzing file...")
        self._log(f"{Icons.FOLDER} {self.input_path}")
        t0 = time.time()

        if not os.path.exists(self.input_path):
            raise FileNotFoundError(self.input_path)

        size_gb = os.path.getsize(self.input_path) / (1024 ** 3)
        self._log(f"{Icons.SAVE} Size: {size_gb:.2f} GB")
        self._msg('progress', 10)

        pf = pq.ParquetFile(self.input_path)
        schema = pf.schema_arrow
        rows = pf.metadata.num_rows

        self._log(f"{Icons.TABLE} {len(schema)} fields, {rows:,} rows")
        self._msg('progress', 30)

        self.df = pl.scan_parquet(self.input_path, n_rows=None, low_memory=False, parallel="auto")
        self._msg('progress', 50)

        fields = []
        schema_dict = dict(zip(schema.names, [str(t) for t in schema.types]))
        num_types = ['int', 'float', 'double', 'Int', 'Float', 'UInt']

        for i, (name, dtype) in enumerate(schema_dict.items()):
            if self._stop:
                return
            try:
                samples = self.df.select(name).head(3).collect()[name].to_list()
            except:
                samples = []
            is_num = any(t in dtype for t in num_types)
            fields.append(FieldInfo(name, dtype, 0, samples, False, is_num))
            self._msg('progress', 50 + int((i + 1) / len(schema_dict) * 45))
            icon = Icons.NUMBER if is_num else Icons.TEXT
            self._log(f"  {icon} {name} ({dtype})")

        self._log(f"{Icons.SUCCESS} Analysis complete in {time.time() - t0:.2f}s")
        self._msg('stats', {'size': size_gb, 'rows': rows, 'fields': len(fields),
                            'numeric': sum(1 for f in fields if f.is_numeric)})
        self._msg('fields', fields)
        self._msg('progress', 100)
        self._msg('done', (True, "Analysis complete"))

    def _execute(self):
        if self.df is None:
            raise ValueError("No data loaded")
        self._log(f"{Icons.CALC} Executing: {self.result_name}")
        self._log(f"   Expression: {self.expr}")
        t0 = time.time()
        self._msg('progress', 20)

        self.df = ExpressionEngine.execute(self.df, self.expr, self.result_name)
        self._msg('progress', 70)

        try:
            sample = self.df.select(self.result_name).head(5).collect()
            for v in sample[self.result_name].to_list():
                self._log(f"   → {v}")
        except Exception as e:
            self._log(f"   {Icons.WARNING} Sample error: {e}")

        self._msg('progress', 100)
        self._log(f"{Icons.SUCCESS} Calculation complete in {time.time() - t0:.2f}s")
        self._msg('new_field', FieldInfo(self.result_name, "Float64", 0, [], True, True))
        self._msg('done', (True, f"Created field: {self.result_name}"))

    def _delete(self):
        if self.df is None:
            raise ValueError("No data loaded")
        self._log(f"{Icons.DELETE} Deleting {len(self.del_fields)} field(s)...")
        for i, f in enumerate(self.del_fields):
            self._log(f"   Removing: {f}")
            self.df = self.df.drop(f)
            self._msg('progress', int((i + 1) / len(self.del_fields) * 100))
        self._log(f"{Icons.SUCCESS} Fields deleted successfully")
        self._msg('deleted', self.del_fields.copy())
        self._msg('done', (True, f"Deleted {len(self.del_fields)} field(s)"))

    def _add(self):
        if self.df is None:
            raise ValueError("No data loaded")
        name, dtype, default = self.new_field
        self._log(f"{Icons.ADD} Adding field: {name} ({dtype})")

        type_map = {"Float64": pl.Float64, "Int64": pl.Int64, "Int32": pl.Int32,
                    "Float32": pl.Float32, "String": pl.Utf8, "Boolean": pl.Boolean}
        pt = type_map.get(dtype, pl.Float64)

        try:
            if default and default.lower() != "null":
                if dtype in ["Float64", "Float32"]:
                    v = float(default)
                elif dtype in ["Int64", "Int32"]:
                    v = int(default)
                elif dtype == "Boolean":
                    v = default.lower() in ['true', '1']
                else:
                    v = default
                self.df = self.df.with_columns(pl.lit(v).cast(pt).alias(name))
            else:
                self.df = self.df.with_columns(pl.lit(None).cast(pt).alias(name))
        except:
            self.df = self.df.with_columns(pl.lit(None).cast(pt).alias(name))

        is_num = dtype in ["Float64", "Int64", "Int32", "Float32"]
        self._msg('new_field', FieldInfo(name, dtype, 0, [], True, is_num))
        self._log(f"{Icons.SUCCESS} Field added successfully")
        self._msg('progress', 100)
        self._msg('done', (True, f"Added field: {name}"))

    def _manage_fields(self):
        """Rename and reorder fields - Fixed version"""
        if self.df is None:
            raise ValueError("No data loaded")

        self._log(f"{Icons.EDIT} Managing fields...")
        self._msg('progress', 10)

        # Get current column names
        current_cols = self.df.collect_schema().names()
        self._log(f"   Current columns: {current_cols[:5]}...")

        # Step 1: Rename (if any)
        if self.rename_map and len(self.rename_map) > 0:
            self._log(f"   Renaming {len(self.rename_map)} field(s)...")
            for old_name, new_name in self.rename_map.items():
                self._log(f"      {old_name} → {new_name}")
            # polars LazyFrame rename method
            self.df = self.df.rename(self.rename_map)
            self._msg('progress', 40)

            # Verify rename
            new_cols = self.df.collect_schema().names()
            self._log(f"   After rename: {new_cols[:5]}...")

        self._msg('progress', 50)

        # Step 2: Reorder (if any)
        if self.new_order and len(self.new_order) > 0:
            self._log(f"   Reordering to: {self.new_order[:5]}...")
            self.df = self.df.select(self.new_order)
            self._msg('progress', 80)

            # Verify order
            final_cols = self.df.collect_schema().names()
            self._log(f"   Final order: {final_cols[:5]}...")

        self._msg('progress', 100)
        self._log(f"{Icons.SUCCESS} Fields updated successfully")

        # Return result to main thread
        self._msg('manage_done', (self.new_order, self.rename_map))
        self._msg('done', (True, "Fields updated"))

    def _save(self):
        if self.df is None:
            raise ValueError("No data loaded")
        self._log(f"{Icons.SAVE} Saving: {self.output_path}")
        t0 = time.time()
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._msg('progress', 10)

        self.df.sink_parquet(self.output_path, compression="zstd",
                             compression_level=3, row_group_size=CHUNK_SIZE)

        size = os.path.getsize(self.output_path) / (1024 ** 3)
        self._log(f"{Icons.SUCCESS} Saved: {size:.2f} GB in {time.time() - t0:.2f}s")
        self._msg('progress', 100)
        self._msg('done', (True, f"Saved to {self.output_path}"))

    def _preview(self):
        if self.df is None:
            raise ValueError("No data loaded")
        self._log(f"{Icons.EYE} Generating preview...")
        df = self.df.head(100).collect()
        self._msg('preview', df)
        self._msg('done', (True, "Preview ready"))


# ============================================================================
# Field Management Dialog - Rewritten Version
# ============================================================================

class ManageFieldsDialog(tk.Toplevel):
    """Field management: sorting and renaming - Completely rewritten"""

    def __init__(self, parent, fields: List[FieldInfo]):
        super().__init__(parent)
        self.result = None
        self.parent = parent

        # Save field info: [(display_name, original_name, is_numeric), ...]
        # display_name = current name (may have been renamed)
        # original_name = name when dialog opened
        self.field_data = []
        for f in fields:
            self.field_data.append([f.name, f.name, f.is_numeric])

        # Window settings - Larger window
        self.title("Manage Fields - Reorder & Rename")
        self.geometry("950x900")
        self.configure(bg=Colors.BG_CARD)
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()

        # Center
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 950) // 2
        y = (self.winfo_screenheight() - 900) // 2
        self.geometry(f"+{x}+{y}")

        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        # Use Canvas for overall scrolling
        outer_frame = tk.Frame(self, bg=Colors.BG_CARD)
        outer_frame.pack(fill=tk.BOTH, expand=True)

        # Create Canvas and scrollbars
        canvas = tk.Canvas(outer_frame, bg=Colors.BG_CARD, highlightthickness=0)
        v_scrollbar = tk.Scrollbar(outer_frame, orient=tk.VERTICAL, command=canvas.yview)
        h_scrollbar = tk.Scrollbar(outer_frame, orient=tk.HORIZONTAL, command=canvas.xview)

        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Main content frame
        main = tk.Frame(canvas, bg=Colors.BG_CARD, padx=30, pady=24)
        canvas_window = canvas.create_window((0, 0), window=main, anchor='nw')

        # Title
        title_frame = tk.Frame(main, bg=Colors.BG_CARD)
        title_frame.pack(fill=tk.X, pady=(0, 20))

        tk.Label(title_frame, text=Icons.SORT, font=FONT_ICON_XLARGE,
                 bg=Colors.BG_CARD, fg=Colors.INDIGO).pack(side=tk.LEFT)

        title_text = tk.Frame(title_frame, bg=Colors.BG_CARD)
        title_text.pack(side=tk.LEFT, padx=(16, 0))
        tk.Label(title_text, text="Manage Fields", font=FONT_TITLE,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_PRIMARY).pack(anchor='w')
        tk.Label(title_text, text="Reorder and rename fields", font=FONT_BODY,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_MUTED).pack(anchor='w')

        # Hint
        hint_frame = tk.Frame(main, bg=Colors.BLUE_LIGHT, padx=16, pady=12)
        hint_frame.pack(fill=tk.X, pady=(0, 20))
        tk.Label(hint_frame,
                 text=f"{Icons.INFO}  Select field → Use buttons on right to Move or Rename",
                 font=FONT_BODY, bg=Colors.BLUE_LIGHT, fg=Colors.BLUE_DARK).pack(anchor='w')

        # Main content area
        content = tk.Frame(main, bg=Colors.BG_CARD)
        content.pack(fill=tk.BOTH, expand=True)

        # Left: List - with scrollbar
        list_frame = tk.Frame(content, bg=Colors.BG_CARD)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # List box container
        list_container = tk.Frame(list_frame, bg=Colors.BORDER, padx=2, pady=2)
        list_container.pack(fill=tk.BOTH, expand=True)

        # List box - Taller
        self.listbox = tk.Listbox(
            list_container,
            bg=Colors.BG_INPUT,
            fg=Colors.TEXT_PRIMARY,
            font=FONT_LIST,
            relief=tk.FLAT,
            selectmode=tk.SINGLE,
            activestyle='none',
            highlightthickness=0,
            selectbackground=Colors.BLUE,
            selectforeground=Colors.TEXT_WHITE,
            height=25,  # Increased height
            width=45  # Increased width
        )

        # Vertical scrollbar
        list_v_scroll = tk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.listbox.yview)
        # Horizontal scrollbar
        list_h_scroll = tk.Scrollbar(list_container, orient=tk.HORIZONTAL, command=self.listbox.xview)

        self.listbox.configure(yscrollcommand=list_v_scroll.set, xscrollcommand=list_h_scroll.set)

        # Layout list and scrollbars
        list_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        list_h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right: Button area - scrollable
        btn_outer = tk.Frame(content, bg=Colors.BG_CARD, padx=24)
        btn_outer.pack(side=tk.RIGHT, fill=tk.Y)

        btn_frame = tk.Frame(btn_outer, bg=Colors.BG_CARD)
        btn_frame.pack(fill=tk.Y)

        # === Move Buttons ===
        tk.Label(btn_frame, text="MOVE", font=FONT_BODY_BOLD,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_SECONDARY).pack(anchor='w', pady=(0, 10))

        self._make_btn(btn_frame, f"{Icons.MOVE_UP} Top", self._move_top, Colors.INDIGO)
        self._make_btn(btn_frame, f"{Icons.MOVE_UP} Up", self._move_up, Colors.BLUE)
        self._make_btn(btn_frame, f"{Icons.MOVE_DOWN} Down", self._move_down, Colors.BLUE)
        self._make_btn(btn_frame, f"{Icons.MOVE_DOWN} Bottom", self._move_bottom, Colors.INDIGO)

        # Separator
        tk.Frame(btn_frame, bg=Colors.BORDER, height=2).pack(fill=tk.X, pady=16)

        # === Sort Buttons ===
        tk.Label(btn_frame, text="SORT", font=FONT_BODY_BOLD,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_SECONDARY).pack(anchor='w', pady=(0, 10))

        self._make_btn(btn_frame, "A → Z", self._sort_az, Colors.TEAL)
        self._make_btn(btn_frame, f"{Icons.NUMBER} By Type", self._sort_type, Colors.GREEN)

        # Separator
        tk.Frame(btn_frame, bg=Colors.BORDER, height=2).pack(fill=tk.X, pady=16)

        # === Rename Button - Highlighted ===
        tk.Label(btn_frame, text="RENAME", font=FONT_BODY_BOLD,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_SECONDARY).pack(anchor='w', pady=(0, 10))

        rename_btn = tk.Button(
            btn_frame,
            text=f"{Icons.EDIT}  RENAME",
            command=self._do_rename,
            bg=Colors.ORANGE,
            fg=Colors.TEXT_WHITE,
            font=FONT_BUTTON_LARGE,
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=12,
            width=12
        )
        rename_btn.pack(fill=tk.X, pady=5)

        # Separator
        tk.Frame(btn_frame, bg=Colors.BORDER, height=2).pack(fill=tk.X, pady=16)

        # === Save Button ===
        tk.Label(btn_frame, text="SAVE", font=FONT_BODY_BOLD,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_SECONDARY).pack(anchor='w', pady=(0, 10))

        save_btn = tk.Button(
            btn_frame,
            text=f"{Icons.SAVE}  SAVE",
            command=self._apply,
            bg=Colors.GREEN,
            fg=Colors.TEXT_WHITE,
            font=FONT_BUTTON_LARGE,
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=12,
            width=12
        )
        save_btn.pack(fill=tk.X, pady=5)

        # Bottom button area
        bottom = tk.Frame(main, bg=Colors.BG_CARD)
        bottom.pack(fill=tk.X, pady=(24, 0))

        # Statistics
        self.count_lbl = tk.Label(bottom, text=f"{len(self.field_data)} fields",
                                  font=FONT_BODY, bg=Colors.BG_CARD, fg=Colors.TEXT_MUTED)
        self.count_lbl.pack(side=tk.LEFT)

        self.status_lbl = tk.Label(bottom, text="",
                                   font=FONT_BODY_BOLD, bg=Colors.BG_CARD, fg=Colors.GREEN)
        self.status_lbl.pack(side=tk.LEFT, padx=(20, 0))

        # Cancel button
        cancel_btn = tk.Button(bottom, text="Cancel",
                               command=self.destroy,
                               bg=Colors.BG_MAIN, fg=Colors.TEXT_SECONDARY,
                               font=FONT_BUTTON, relief=tk.FLAT,
                               padx=24, pady=14, cursor='hand2')
        cancel_btn.pack(side=tk.RIGHT)

        # Apply button
        ok_btn = tk.Button(bottom, text=f"{Icons.CHECK} APPLY CHANGES",
                           command=self._apply,
                           bg=Colors.INDIGO, fg=Colors.TEXT_WHITE,
                           font=FONT_BUTTON_LARGE, relief=tk.FLAT,
                           padx=28, pady=14, cursor='hand2')
        ok_btn.pack(side=tk.RIGHT, padx=(0, 16))

        # Update scroll region
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox('all'))
            # Adjust width to fit content
            canvas.itemconfig(canvas_window, width=max(event.width, main.winfo_reqwidth()))

        main.bind('<Configure>', on_configure)

        # Bind mouse wheel
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

        canvas.bind('<Enter>', lambda e: canvas.bind_all('<MouseWheel>', on_mousewheel))
        canvas.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))

    def _make_btn(self, parent, text, command, color):
        """Create regular button"""
        btn = tk.Button(parent, text=text, command=command,
                        bg=color, fg=Colors.TEXT_WHITE,
                        font=FONT_BUTTON, relief=tk.FLAT,
                        padx=16, pady=10, cursor='hand2', width=12)
        btn.pack(pady=4, fill=tk.X)

    def _refresh_list(self, select_idx=None):
        """Refresh list display"""
        self.listbox.delete(0, tk.END)

        renamed_count = 0
        for i, (display_name, original_name, is_numeric) in enumerate(self.field_data):
            icon = Icons.NUMBER if is_numeric else Icons.TEXT

            if display_name != original_name:
                # Renamed: show "new_name ← original_name"
                text = f"  {icon}  {display_name}  ←  {original_name}"
                renamed_count += 1
            else:
                text = f"  {icon}  {display_name}"

            self.listbox.insert(tk.END, text)

        # Restore selection
        if select_idx is not None and 0 <= select_idx < len(self.field_data):
            self.listbox.selection_set(select_idx)
            self.listbox.see(select_idx)
        elif self.field_data:
            self.listbox.selection_set(0)

        # Update status
        self.count_lbl.configure(text=f"{len(self.field_data)} fields")
        if renamed_count > 0:
            self.status_lbl.configure(text=f"{renamed_count} field(s) will be renamed")
        else:
            self.status_lbl.configure(text="")

    def _get_selected_index(self):
        """Get currently selected index"""
        sel = self.listbox.curselection()
        if sel:
            return sel[0]
        return None

    # === Move Operations ===
    def _move_up(self):
        idx = self._get_selected_index()
        if idx is not None and idx > 0:
            self.field_data[idx], self.field_data[idx - 1] = self.field_data[idx - 1], self.field_data[idx]
            self._refresh_list(idx - 1)

    def _move_down(self):
        idx = self._get_selected_index()
        if idx is not None and idx < len(self.field_data) - 1:
            self.field_data[idx], self.field_data[idx + 1] = self.field_data[idx + 1], self.field_data[idx]
            self._refresh_list(idx + 1)

    def _move_top(self):
        idx = self._get_selected_index()
        if idx is not None and idx > 0:
            item = self.field_data.pop(idx)
            self.field_data.insert(0, item)
            self._refresh_list(0)

    def _move_bottom(self):
        idx = self._get_selected_index()
        if idx is not None and idx < len(self.field_data) - 1:
            item = self.field_data.pop(idx)
            self.field_data.append(item)
            self._refresh_list(len(self.field_data) - 1)

    # === Sort Operations ===
    def _sort_az(self):
        idx = self._get_selected_index()
        selected_item = self.field_data[idx] if idx is not None else None

        self.field_data.sort(key=lambda x: x[0].lower())  # Sort by display name

        new_idx = 0
        if selected_item:
            try:
                new_idx = self.field_data.index(selected_item)
            except:
                pass
        self._refresh_list(new_idx)

    def _sort_type(self):
        idx = self._get_selected_index()
        selected_item = self.field_data[idx] if idx is not None else None

        # Numeric types first, then by name
        self.field_data.sort(key=lambda x: (not x[2], x[0].lower()))

        new_idx = 0
        if selected_item:
            try:
                new_idx = self.field_data.index(selected_item)
            except:
                pass
        self._refresh_list(new_idx)

    # === Rename Operation ===
    def _do_rename(self):
        """Execute rename"""
        idx = self._get_selected_index()
        if idx is None:
            messagebox.showinfo("Info", "Please select a field first!", parent=self)
            return

        current_name = self.field_data[idx][0]

        # Get all current names (excluding self)
        existing_names = set(item[0] for i, item in enumerate(self.field_data) if i != idx)

        # Show simple input dialog
        new_name = simpledialog.askstring(
            "Rename Field",
            f"Enter new name for '{current_name}':",
            initialvalue=current_name,
            parent=self
        )

        if new_name is None:
            return  # User cancelled

        new_name = new_name.strip()

        if not new_name:
            messagebox.showwarning("Error", "Name cannot be empty!", parent=self)
            return

        if new_name in existing_names:
            messagebox.showwarning("Error", f"Name '{new_name}' already exists!", parent=self)
            return

        # Update display name
        self.field_data[idx][0] = new_name
        self._refresh_list(idx)

    # === Apply Changes ===
    def _apply(self):
        """Apply all changes"""
        # Build new order (using final display names)
        new_order = [item[0] for item in self.field_data]

        # Build rename map {original_name: new_name}
        rename_map = {}
        for display_name, original_name, _ in self.field_data:
            if display_name != original_name:
                rename_map[original_name] = display_name

        print(f"DEBUG: new_order = {new_order}")
        print(f"DEBUG: rename_map = {rename_map}")

        self.result = (new_order, rename_map)
        self.destroy()


# ============================================================================
# Add Field Dialog
# ============================================================================

class AddFieldDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.result = None

        self.title(f"{Icons.ADD} Add New Field")
        self.geometry("480x380")
        self.configure(bg=Colors.BG_CARD)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        main = tk.Frame(self, bg=Colors.BG_CARD, padx=32, pady=28)
        main.pack(fill=tk.BOTH, expand=True)

        # Title
        title_frame = tk.Frame(main, bg=Colors.BG_CARD)
        title_frame.pack(fill=tk.X, pady=(0, 24))
        tk.Label(title_frame, text=Icons.ADD, font=FONT_ICON_XLARGE,
                 bg=Colors.BG_CARD, fg=Colors.GREEN).pack(side=tk.LEFT)
        tk.Label(title_frame, text="Add New Field", font=FONT_TITLE,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_PRIMARY).pack(side=tk.LEFT, padx=(14, 0))

        # Name
        tk.Label(main, text="Field Name", font=FONT_BODY_BOLD,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_SECONDARY).pack(anchor='w')
        self.name_e = tk.Entry(main, bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
                               relief=tk.FLAT, font=FONT_BODY,
                               highlightthickness=2, highlightbackground=Colors.BORDER,
                               highlightcolor=Colors.BORDER_FOCUS)
        self.name_e.pack(fill=tk.X, ipady=12, pady=(6, 18))

        # Type
        tk.Label(main, text="Data Type", font=FONT_BODY_BOLD,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_SECONDARY).pack(anchor='w')
        self.type_var = tk.StringVar(value="Float64")
        type_combo = ttk.Combobox(main, textvariable=self.type_var, state='readonly',
                                  values=["Float64", "Int64", "Int32", "Float32", "String", "Boolean"],
                                  font=FONT_BODY)
        type_combo.pack(fill=tk.X, pady=(6, 18))

        # Default value
        tk.Label(main, text="Default Value", font=FONT_BODY_BOLD,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_SECONDARY).pack(anchor='w')
        self.default_e = tk.Entry(main, bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
                                  relief=tk.FLAT, font=FONT_BODY,
                                  highlightthickness=2, highlightbackground=Colors.BORDER,
                                  highlightcolor=Colors.BORDER_FOCUS)
        self.default_e.pack(fill=tk.X, ipady=12, pady=(6, 28))
        self.default_e.insert(0, "null")

        # Buttons
        btn_frame = tk.Frame(main, bg=Colors.BG_CARD)
        btn_frame.pack(fill=tk.X)

        cancel_btn = tk.Button(btn_frame, text="Cancel", command=self.destroy,
                               bg=Colors.BG_MAIN, fg=Colors.TEXT_SECONDARY,
                               font=FONT_BUTTON, relief=tk.FLAT,
                               padx=18, pady=10, cursor='hand2')
        cancel_btn.pack(side=tk.RIGHT)

        ok_btn = tk.Button(btn_frame, text=f"{Icons.ADD} Add", command=self._ok,
                           bg=Colors.GREEN, fg=Colors.TEXT_WHITE,
                           font=FONT_BUTTON, relief=tk.FLAT,
                           padx=24, pady=10, cursor='hand2')
        ok_btn.pack(side=tk.RIGHT, padx=(0, 14))

        # Center
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 480) // 2
        y = (self.winfo_screenheight() - 380) // 2
        self.geometry(f"+{x}+{y}")

    def _ok(self):
        name = self.name_e.get().strip()
        if not name:
            messagebox.showwarning("Error", "Please enter a field name")
            return
        self.result = (name, self.type_var.get(), self.default_e.get().strip())
        self.destroy()


# ============================================================================
# Common Components
# ============================================================================

class Card(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=Colors.BG_CARD, highlightbackground=Colors.BORDER,
                         highlightthickness=1, **kwargs)


class IconButton(tk.Frame):
    def __init__(self, parent, icon, text, command, color=Colors.BLUE, **kwargs):
        super().__init__(parent, bg=color, cursor='hand2', **kwargs)

        self.base_color = color
        self.command = command
        self.enabled = True

        self.icon_lbl = tk.Label(self, text=icon, font=FONT_ICON,
                                 bg=color, fg=Colors.TEXT_WHITE)
        self.icon_lbl.pack(side=tk.LEFT, padx=(16, 6), pady=14)

        self.text_lbl = tk.Label(self, text=text, font=FONT_BUTTON,
                                 bg=color, fg=Colors.TEXT_WHITE)
        self.text_lbl.pack(side=tk.LEFT, padx=(0, 16), pady=14)

        for w in [self, self.icon_lbl, self.text_lbl]:
            w.bind('<Enter>', self._enter)
            w.bind('<Leave>', self._leave)
            w.bind('<Button-1>', self._click)

    def _darken(self, c):
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        return f'#{int(r * 0.85):02x}{int(g * 0.85):02x}{int(b * 0.85):02x}'

    def _enter(self, e):
        if self.enabled:
            color = self._darken(self.base_color)
            for w in [self, self.icon_lbl, self.text_lbl]:
                w.configure(bg=color)

    def _leave(self, e):
        color = self.base_color if self.enabled else Colors.TEXT_MUTED
        for w in [self, self.icon_lbl, self.text_lbl]:
            w.configure(bg=color)

    def _click(self, e):
        if self.enabled and self.command:
            self.command()

    def set_enabled(self, enabled):
        self.enabled = enabled
        color = self.base_color if enabled else Colors.TEXT_MUTED
        for w in [self, self.icon_lbl, self.text_lbl]:
            w.configure(bg=color)
        self.configure(cursor='hand2' if enabled else 'arrow')


class FieldButton(tk.Frame):
    def __init__(self, parent, field: FieldInfo, on_insert, on_select, **kwargs):
        bg = Colors.BTN_FIELD_NUM if field.is_numeric else Colors.BTN_FIELD_STR
        border_color = Colors.GREEN if field.is_numeric else Colors.PURPLE

        super().__init__(parent, bg=bg, cursor='hand2',
                         highlightbackground=border_color,
                         highlightthickness=2, **kwargs)

        self.field = field
        self.base_bg = bg
        self.border_color = border_color
        self.on_insert = on_insert
        self.on_select = on_select
        self.selected = False

        icon = Icons.NUMBER if field.is_numeric else Icons.TEXT
        icon_color = Colors.GREEN if field.is_numeric else Colors.PURPLE

        content = tk.Frame(self, bg=bg)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # First row: checkbox + icon
        row1 = tk.Frame(content, bg=bg)
        row1.pack(fill=tk.X)

        self.check_lbl = tk.Label(row1, text="○", font=('Segoe UI', 14),
                                  bg=bg, fg=Colors.TEXT_MUTED)
        self.check_lbl.pack(side=tk.LEFT)

        self.icon_lbl = tk.Label(row1, text=icon, font=('Segoe UI Emoji', 14),
                                 bg=bg, fg=icon_color)
        self.icon_lbl.pack(side=tk.LEFT, padx=(6, 0))

        if field.is_new:
            badge = tk.Label(row1, text=" NEW ", font=('Segoe UI', 9, 'bold'),
                             bg=Colors.GREEN, fg=Colors.TEXT_WHITE)
            badge.pack(side=tk.RIGHT)

        # Second row: full field name (wrappable)
        self.name_lbl = tk.Label(content, text=field.name, font=FONT_BODY_BOLD,
                                 bg=bg, fg=Colors.TEXT_PRIMARY, anchor='w',
                                 justify=tk.LEFT, wraplength=280)
        self.name_lbl.pack(fill=tk.X, pady=(6, 0))

        # Third row: data type
        self.type_lbl = tk.Label(content, text=field.dtype, font=FONT_TINY,
                                 bg=bg, fg=Colors.TEXT_MUTED, anchor='w')
        self.type_lbl.pack(fill=tk.X, pady=(4, 0))

        widgets = [self, content, row1, self.check_lbl, self.icon_lbl,
                   self.name_lbl, self.type_lbl]
        for w in widgets:
            w.bind('<Button-1>', self._on_left_click)
            w.bind('<Button-3>', self._on_right_click)
            w.bind('<Enter>', self._enter)
            w.bind('<Leave>', self._leave)

    def _on_left_click(self, e):
        self.selected = not self.selected
        self._update_visual()
        self.on_select(self.field.name, self.selected)

    def _on_right_click(self, e):
        self.on_insert(self.field.name)

    def _enter(self, e):
        if not self.selected:
            self._set_bg(Colors.BG_HOVER)

    def _leave(self, e):
        self._update_visual()

    def _update_visual(self):
        if self.selected:
            self._set_bg(Colors.BG_SELECTED)
            self.configure(highlightbackground=Colors.BLUE)
            self.check_lbl.configure(text="●", fg=Colors.BLUE)
        else:
            self._set_bg(self.base_bg)
            self.configure(highlightbackground=self.border_color)
            self.check_lbl.configure(text="○", fg=Colors.TEXT_MUTED)

    def _set_bg(self, color):
        for w in [self, self.check_lbl, self.icon_lbl, self.name_lbl, self.type_lbl]:
            try:
                w.configure(bg=color)
            except:
                pass
        for child in self.winfo_children():
            try:
                child.configure(bg=color)
                for subchild in child.winfo_children():
                    try:
                        subchild.configure(bg=color)
                    except:
                        pass
            except:
                pass

    def set_selected(self, selected: bool):
        self.selected = selected
        self._update_visual()

    def update_name(self, new_name: str):
        """Update field name display"""
        self.field.name = new_name
        self.name_lbl.configure(text=new_name)


class CalcBtn(tk.Label):
    def __init__(self, parent, text, cmd, bg=Colors.BTN_NUMBER, fg=Colors.TEXT_PRIMARY,
                 width=4, font_size=15, **kwargs):
        super().__init__(parent, text=text, bg=bg, fg=fg,
                         font=('Segoe UI', font_size, 'bold'),
                         width=width, pady=14, cursor='hand2', **kwargs)
        self.base_bg = bg
        self.cmd = cmd
        self.bind('<Enter>', lambda e: self.configure(bg=Colors.BG_HOVER))
        self.bind('<Leave>', lambda e: self.configure(bg=self.base_bg))
        self.bind('<Button-1>', lambda e: self.cmd() if self.cmd else None)


# ============================================================================
# Main Application
# ============================================================================

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1800x1050")
        self.root.configure(bg=Colors.BG_MAIN)
        self.root.minsize(1600, 900)

        self.fields: List[FieldInfo] = []
        self.selected: set = set()
        self.field_btns: Dict[str, FieldButton] = {}
        self.stats = {}

        self.q = queue.Queue()
        self.worker = Worker(self.q)
        self.worker.start()

        self.t0 = None
        self.timer_on = False

        self._setup_styles()
        self._build_ui()
        self._defaults()
        self._poll()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('TCombobox',
                        fieldbackground=Colors.BG_INPUT,
                        background=Colors.BG_INPUT,
                        foreground=Colors.TEXT_PRIMARY,
                        arrowcolor=Colors.TEXT_SECONDARY)

        style.configure('Custom.Horizontal.TProgressbar',
                        troughcolor=Colors.BG_MAIN,
                        background=Colors.BLUE,
                        lightcolor=Colors.BLUE,
                        darkcolor=Colors.BLUE_DARK,
                        bordercolor=Colors.BORDER,
                        thickness=12)

    def _build_ui(self):
        main = tk.Frame(self.root, bg=Colors.BG_MAIN)
        main.pack(fill=tk.BOTH, expand=True, padx=26, pady=20)

        self._build_header(main)

        content = tk.Frame(main, bg=Colors.BG_MAIN)
        content.pack(fill=tk.BOTH, expand=True, pady=(20, 0))

        # Left column
        left = tk.Frame(content, bg=Colors.BG_MAIN)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14))

        self._build_file_panel(left)
        self._build_fields_panel(left)

        # Middle column - Calculator
        middle = tk.Frame(content, bg=Colors.BG_MAIN, width=450)
        middle.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        middle.pack_propagate(False)

        self._build_calc_panel(middle)

        # Right column - Log and preview
        right = tk.Frame(content, bg=Colors.BG_MAIN, width=480)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        self._build_log_panel(right)
        self._build_preview_panel(right)

        self._build_status(main)

    def _build_header(self, parent):
        card = Card(parent)
        card.pack(fill=tk.X)

        inner = tk.Frame(card, bg=Colors.BG_CARD, padx=26, pady=20)
        inner.pack(fill=tk.X)

        left = tk.Frame(inner, bg=Colors.BG_CARD)
        left.pack(side=tk.LEFT)

        logo = tk.Frame(left, bg=Colors.BLUE, padx=18, pady=14)
        logo.pack(side=tk.LEFT, padx=(0, 20))
        tk.Label(logo, text=Icons.TABLE, font=FONT_ICON_XLARGE,
                 bg=Colors.BLUE, fg=Colors.TEXT_WHITE).pack()

        title_frame = tk.Frame(left, bg=Colors.BG_CARD)
        title_frame.pack(side=tk.LEFT)

        tk.Label(title_frame, text=APP_NAME, font=FONT_TITLE,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_PRIMARY).pack(anchor='w')

        tk.Label(title_frame,
                 text=f"{Icons.SPARKLE} Expression Builder • Field Management",
                 font=FONT_BODY, bg=Colors.BG_CARD, fg=Colors.TEXT_MUTED).pack(anchor='w', pady=(6, 0))

        stats_frame = tk.Frame(inner, bg=Colors.BG_CARD)
        stats_frame.pack(side=tk.RIGHT)

        self._create_stat_badge(stats_frame, Icons.TABLE, "Rows", "0", 'rows',
                                Colors.BLUE_LIGHT, Colors.BLUE)
        self._create_stat_badge(stats_frame, Icons.FIELD, "Fields", "0", 'fields',
                                Colors.GREEN_LIGHT, Colors.GREEN)
        self._create_stat_badge(stats_frame, Icons.SAVE, "Size", "0 GB", 'size',
                                Colors.PURPLE_LIGHT, Colors.PURPLE)
        self._create_stat_badge(stats_frame, Icons.NUMBER, "Numeric", "0", 'num',
                                Colors.ORANGE_LIGHT, Colors.ORANGE)

    def _create_stat_badge(self, parent, icon, label, value, key, bg, fg):
        frame = tk.Frame(parent, bg=bg, padx=20, pady=14)
        frame.pack(side=tk.LEFT, padx=8)

        tk.Label(frame, text=icon, font=FONT_ICON,
                 bg=bg, fg=fg).pack(side=tk.LEFT)

        inner = tk.Frame(frame, bg=bg)
        inner.pack(side=tk.LEFT, padx=(14, 0))

        tk.Label(inner, text=label, font=FONT_TINY,
                 bg=bg, fg=Colors.TEXT_MUTED).pack(anchor='w')

        lbl = tk.Label(inner, text=value, font=FONT_SUBHEADING,
                       bg=bg, fg=fg)
        lbl.pack(anchor='w')
        setattr(self, f'stat_{key}', lbl)

    def _build_file_panel(self, parent):
        card = Card(parent)
        card.pack(fill=tk.X)

        inner = tk.Frame(card, bg=Colors.BG_CARD, padx=22, pady=20)
        inner.pack(fill=tk.X)

        title_row = tk.Frame(inner, bg=Colors.BG_CARD)
        title_row.pack(fill=tk.X, pady=(0, 18))

        tk.Label(title_row, text=Icons.FOLDER_OPEN, font=FONT_ICON,
                 bg=Colors.BG_CARD).pack(side=tk.LEFT)
        tk.Label(title_row, text="File Input / Output", font=FONT_HEADING,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_HEADING).pack(side=tk.LEFT, padx=(14, 0))

        self._create_file_row(inner, "Input", 'input')
        self._create_file_row(inner, "Output", 'output')

        btn_row = tk.Frame(inner, bg=Colors.BG_CARD)
        btn_row.pack(fill=tk.X, pady=(20, 0))

        self.analyze_btn = IconButton(btn_row, Icons.PLAY, "Analyze",
                                      self._analyze, Colors.GREEN)
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 14))

        self.stop_btn = IconButton(btn_row, Icons.STOP, "Stop",
                                   self._stop, Colors.RED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 14))

        self.save_btn = IconButton(btn_row, Icons.SAVE, "Save",
                                   self._save, Colors.BLUE)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 14))

        self.preview_btn = IconButton(btn_row, Icons.EYE, "Preview",
                                      self._preview, Colors.CYAN)
        self.preview_btn.pack(side=tk.LEFT)

        prog_frame = tk.Frame(inner, bg=Colors.BG_CARD)
        prog_frame.pack(fill=tk.X, pady=(20, 0))

        self.prog_var = tk.DoubleVar()
        self.prog_bar = ttk.Progressbar(prog_frame, variable=self.prog_var,
                                        maximum=100, style='Custom.Horizontal.TProgressbar')
        self.prog_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        self.prog_lbl = tk.Label(prog_frame, text="0%", font=FONT_BODY_BOLD,
                                 bg=Colors.BG_CARD, fg=Colors.BLUE, width=6)
        self.prog_lbl.pack(side=tk.LEFT, padx=(16, 0))

        self.time_lbl = tk.Label(prog_frame, text=f"{Icons.CLOCK} 00:00:00",
                                 font=FONT_BODY,
                                 bg=Colors.BG_CARD, fg=Colors.TEXT_MUTED, width=14)
        self.time_lbl.pack(side=tk.LEFT)

    def _create_file_row(self, parent, label, key):
        row = tk.Frame(parent, bg=Colors.BG_CARD)
        row.pack(fill=tk.X, pady=8)

        tk.Label(row, text=f"{label}:", font=FONT_BODY_BOLD,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_SECONDARY, width=8, anchor='w').pack(side=tk.LEFT)

        entry = tk.Entry(row, bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
                         relief=tk.FLAT, font=FONT_BODY,
                         highlightthickness=2, highlightbackground=Colors.BORDER,
                         highlightcolor=Colors.BORDER_FOCUS)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=16, ipady=14)
        setattr(self, f'{key}_e', entry)

        browse_cmd = self._browse_in if key == 'input' else self._browse_out
        browse_btn = tk.Label(row, text=f"{Icons.FOLDER} Browse", bg=Colors.BG_MAIN,
                              fg=Colors.TEXT_SECONDARY, font=FONT_SMALL, padx=14, pady=8,
                              cursor='hand2')
        browse_btn.pack(side=tk.LEFT)
        browse_btn.bind('<Button-1>', lambda e, c=browse_cmd: c())

    def _build_fields_panel(self, parent):
        card = Card(parent)
        card.pack(fill=tk.BOTH, expand=True, pady=(16, 0))

        inner = tk.Frame(card, bg=Colors.BG_CARD, padx=20, pady=18)
        inner.pack(fill=tk.BOTH, expand=True)

        header = tk.Frame(inner, bg=Colors.BG_CARD)
        header.pack(fill=tk.X, pady=(0, 14))

        tk.Label(header, text=Icons.FIELD, font=FONT_ICON,
                 bg=Colors.BG_CARD).pack(side=tk.LEFT)
        tk.Label(header, text="Fields", font=FONT_HEADING,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_HEADING).pack(side=tk.LEFT, padx=(12, 0))

        btn_frame = tk.Frame(header, bg=Colors.BG_CARD)
        btn_frame.pack(side=tk.RIGHT)

        # Add button
        self.add_btn = tk.Button(btn_frame, text=f"{Icons.ADD} Add", command=self._add_field,
                                 bg=Colors.GREEN, fg=Colors.TEXT_WHITE, relief=tk.FLAT,
                                 font=FONT_SMALL_BOLD, padx=14, pady=8,
                                 cursor='hand2', state=tk.DISABLED)
        self.add_btn.pack(side=tk.LEFT, padx=5)

        # Manage button (sort + rename)
        self.manage_btn = tk.Button(btn_frame, text=f"{Icons.SORT} Manage", command=self._manage_fields,
                                    bg=Colors.INDIGO, fg=Colors.TEXT_WHITE, relief=tk.FLAT,
                                    font=FONT_SMALL_BOLD, padx=14, pady=8,
                                    cursor='hand2', state=tk.DISABLED)
        self.manage_btn.pack(side=tk.LEFT, padx=5)

        # Delete button
        self.del_btn = tk.Button(btn_frame, text=f"{Icons.DELETE} Delete", command=self._del_fields,
                                 bg=Colors.RED, fg=Colors.TEXT_WHITE, relief=tk.FLAT,
                                 font=FONT_SMALL_BOLD, padx=14, pady=8,
                                 cursor='hand2', state=tk.DISABLED)
        self.del_btn.pack(side=tk.LEFT, padx=5)

        tk.Frame(btn_frame, bg=Colors.BORDER, width=2).pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=4)

        self.sel_all_btn = tk.Button(btn_frame, text="☑ All", command=self._select_all,
                                     bg=Colors.BG_MAIN, fg=Colors.TEXT_SECONDARY, relief=tk.FLAT,
                                     font=FONT_TINY, padx=12, pady=6,
                                     cursor='hand2', state=tk.DISABLED)
        self.sel_all_btn.pack(side=tk.LEFT, padx=4)

        self.desel_btn = tk.Button(btn_frame, text="☐ None", command=self._deselect_all,
                                   bg=Colors.BG_MAIN, fg=Colors.TEXT_SECONDARY, relief=tk.FLAT,
                                   font=FONT_TINY, padx=12, pady=6,
                                   cursor='hand2', state=tk.DISABLED)
        self.desel_btn.pack(side=tk.LEFT, padx=4)

        hint_frame = tk.Frame(inner, bg=Colors.BLUE_LIGHT, padx=16, pady=12)
        hint_frame.pack(fill=tk.X, pady=(0, 14))

        tk.Label(hint_frame,
                 text=f"{Icons.INFO} Left-click: Select • Right-click: Insert [field] • Manage: Reorder & Rename",
                 font=FONT_BODY, bg=Colors.BLUE_LIGHT, fg=Colors.BLUE_DARK).pack(anchor='w')

        container = tk.Frame(inner, bg=Colors.BG_MAIN)
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(container, bg=Colors.BG_MAIN, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient=tk.VERTICAL, command=self.canvas.yview)

        self.fields_frame = tk.Frame(self.canvas, bg=Colors.BG_MAIN)

        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_win = self.canvas.create_window((0, 0), window=self.fields_frame, anchor='nw')

        self.fields_frame.bind('<Configure>',
                               lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>',
                         lambda e: self.canvas.itemconfig(self.canvas_win, width=e.width))

        self.canvas.bind('<Enter>', lambda e: self.canvas.bind_all('<MouseWheel>', self._scroll))
        self.canvas.bind('<Leave>', lambda e: self.canvas.unbind_all('<MouseWheel>'))

    def _scroll(self, e):
        self.canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')

    def _build_calc_panel(self, parent):
        card = Card(parent)
        card.pack(fill=tk.BOTH, expand=True)

        inner = tk.Frame(card, bg=Colors.BG_CARD, padx=20, pady=20)
        inner.pack(fill=tk.BOTH, expand=True)

        title = tk.Frame(inner, bg=Colors.BG_CARD)
        title.pack(fill=tk.X, pady=(0, 18))

        tk.Label(title, text=Icons.CALC, font=FONT_ICON,
                 bg=Colors.BG_CARD).pack(side=tk.LEFT)
        tk.Label(title, text="Expression Calculator", font=FONT_HEADING,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_HEADING).pack(side=tk.LEFT, padx=(14, 0))

        tk.Label(inner, text="Output Field Name", font=FONT_BODY_BOLD,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_SECONDARY).pack(anchor='w')

        self.result_e = tk.Entry(inner, bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
                                 relief=tk.FLAT, font=FONT_BODY,
                                 highlightthickness=2, highlightbackground=Colors.BORDER,
                                 highlightcolor=Colors.BORDER_FOCUS)
        self.result_e.pack(fill=tk.X, ipady=12, pady=(8, 18))
        self.result_e.insert(0, "calculated_field")

        expr_container = tk.Frame(inner, bg=Colors.BG_DARK, padx=18, pady=16)
        expr_container.pack(fill=tk.X, pady=(0, 14))

        expr_header = tk.Frame(expr_container, bg=Colors.BG_DARK)
        expr_header.pack(fill=tk.X, pady=(0, 10))

        tk.Label(expr_header, text=f"{Icons.LIGHTNING} Expression", font=FONT_SMALL_BOLD,
                 bg=Colors.BG_DARK, fg=Colors.TEXT_MUTED).pack(side=tk.LEFT)

        self.expr_text = tk.Text(expr_container, bg=Colors.BG_DARK, fg=Colors.TEXT_CODE,
                                 font=FONT_CODE, height=4, relief=tk.FLAT,
                                 insertbackground=Colors.TEXT_CODE, wrap=tk.WORD,
                                 padx=10, pady=10)
        self.expr_text.pack(fill=tk.X)
        self.expr_text.insert('1.0', "# Right-click fields to insert\n# Example: ([A] + [B]) * 100")
        self.expr_text.bind('<KeyRelease>', lambda e: self._validate())

        self.valid_lbl = tk.Label(inner, text="", font=FONT_SMALL,
                                  bg=Colors.BG_CARD, fg=Colors.GREEN)
        self.valid_lbl.pack(anchor='w', pady=(0, 14))

        self._create_section_label(inner, "Operators")
        ops_frame = tk.Frame(inner, bg=Colors.BG_CARD)
        ops_frame.pack(fill=tk.X, pady=(0, 14))

        ops = [('+', '+'), ('−', '-'), ('×', '*'), ('÷', '/'),
               ('%', '%'), ('^', '^'), ('(', '('), (')', ')')]
        for txt, ins in ops:
            CalcBtn(ops_frame, txt, lambda i=ins: self._ins(i),
                    Colors.BTN_OPERATOR, Colors.BLUE, width=3, font_size=16).pack(side=tk.LEFT, padx=4, pady=4)

        self._create_section_label(inner, "Numbers")
        nums_frame = tk.Frame(inner, bg=Colors.BG_CARD)
        nums_frame.pack(fill=tk.X, pady=(0, 14))

        for num in ['7', '8', '9', '4', '5', '6', '1', '2', '3', '0', '.', ',']:
            CalcBtn(nums_frame, num, lambda n=num: self._ins(n),
                    Colors.BTN_NUMBER, Colors.TEXT_PRIMARY, width=3, font_size=15).pack(side=tk.LEFT, padx=4, pady=4)

        self._create_section_label(inner, "Functions")
        func_frame = tk.Frame(inner, bg=Colors.BG_CARD)
        func_frame.pack(fill=tk.X, pady=(0, 18))

        funcs = [('abs', 'abs()'), ('sqrt', 'sqrt()'), ('log', 'log()'), ('log10', 'log10()'),
                 ('exp', 'exp()'), ('sin', 'sin()'), ('cos', 'cos()'), ('tan', 'tan()'),
                 ('ceil', 'ceil()'), ('floor', 'floor()'), ('round', 'round()'), ('sign', 'sign()')]

        for i, (txt, ins) in enumerate(funcs):
            row, col = divmod(i, 4)
            btn = CalcBtn(func_frame, txt, lambda s=ins: self._ins_func(s),
                          Colors.BTN_FUNCTION, Colors.ORANGE, width=7, font_size=13)
            btn.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
        for c in range(4):
            func_frame.columnconfigure(c, weight=1)

        action_frame = tk.Frame(inner, bg=Colors.BG_CARD)
        action_frame.pack(fill=tk.X, pady=(12, 0))

        clear_btn = tk.Button(action_frame, text=f"⌫ Clear", command=self._clear_expr,
                              bg=Colors.ORANGE, fg=Colors.TEXT_WHITE, relief=tk.FLAT,
                              font=FONT_BUTTON, padx=20, pady=14,
                              cursor='hand2')
        clear_btn.pack(side=tk.LEFT, padx=(0, 14))

        self.exec_btn = tk.Button(action_frame, text=f"= Execute", command=self._execute,
                                  bg=Colors.BLUE, fg=Colors.TEXT_WHITE, relief=tk.FLAT,
                                  font=FONT_BUTTON_LARGE, padx=30, pady=14,
                                  cursor='hand2', state=tk.DISABLED)
        self.exec_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _create_section_label(self, parent, text):
        tk.Label(parent, text=text, font=FONT_SMALL_BOLD,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_SECONDARY).pack(anchor='w', pady=(10, 6))

    def _build_log_panel(self, parent):
        card = Card(parent)
        card.pack(fill=tk.BOTH, expand=True)

        inner = tk.Frame(card, bg=Colors.BG_CARD, padx=18, pady=18)
        inner.pack(fill=tk.BOTH, expand=True)

        header = tk.Frame(inner, bg=Colors.BG_CARD)
        header.pack(fill=tk.X, pady=(0, 14))

        tk.Label(header, text=Icons.TEXT, font=FONT_ICON,
                 bg=Colors.BG_CARD).pack(side=tk.LEFT)
        tk.Label(header, text="Processing Log", font=FONT_SUBHEADING,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_HEADING).pack(side=tk.LEFT, padx=(12, 0))

        clear_btn = tk.Label(header, text="Clear", bg=Colors.BG_MAIN, fg=Colors.TEXT_SECONDARY,
                             font=FONT_SMALL, padx=12, pady=6, cursor='hand2')
        clear_btn.pack(side=tk.RIGHT)
        clear_btn.bind('<Button-1>', lambda e: self._clear_log())

        self.autoscroll = tk.BooleanVar(value=True)
        tk.Checkbutton(header, text="Auto-scroll", variable=self.autoscroll,
                       bg=Colors.BG_CARD, fg=Colors.TEXT_MUTED,
                       selectcolor=Colors.BG_MAIN, font=FONT_TINY,
                       activebackground=Colors.BG_CARD).pack(side=tk.RIGHT, padx=14)

        self.log = scrolledtext.ScrolledText(inner, bg=Colors.BG_MAIN, fg=Colors.TEXT_PRIMARY,
                                             font=FONT_CODE_SMALL, relief=tk.FLAT, wrap=tk.WORD,
                                             highlightthickness=1, highlightbackground=Colors.BORDER)
        self.log.pack(fill=tk.BOTH, expand=True)

        self.log.tag_configure('error', foreground=Colors.RED)
        self.log.tag_configure('success', foreground=Colors.GREEN)
        self.log.tag_configure('warning', foreground=Colors.ORANGE)
        self.log.tag_configure('info', foreground=Colors.BLUE)

    def _build_preview_panel(self, parent):
        card = Card(parent)
        card.pack(fill=tk.X, pady=(16, 0))

        inner = tk.Frame(card, bg=Colors.BG_CARD, padx=18, pady=18)
        inner.pack(fill=tk.X)

        header = tk.Frame(inner, bg=Colors.BG_CARD)
        header.pack(fill=tk.X, pady=(0, 14))

        tk.Label(header, text=Icons.EYE, font=FONT_ICON,
                 bg=Colors.BG_CARD).pack(side=tk.LEFT)
        tk.Label(header, text="Data Preview", font=FONT_SUBHEADING,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_HEADING).pack(side=tk.LEFT, padx=(12, 0))

        tree_frame = tk.Frame(inner, bg=Colors.BG_CARD)
        tree_frame.pack(fill=tk.X)

        self.tree = ttk.Treeview(tree_frame, height=6, show='headings')
        h_scroll = tk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(xscrollcommand=h_scroll.set)
        self.tree.pack(fill=tk.X)
        h_scroll.pack(fill=tk.X)

    def _build_status(self, parent):
        status = tk.Frame(parent, bg=Colors.BG_CARD, height=40)
        status.pack(fill=tk.X, pady=(16, 0))

        self.status_lbl = tk.Label(status, text=f"{Icons.CHECK} Ready", font=FONT_BODY,
                                   bg=Colors.BG_CARD, fg=Colors.TEXT_MUTED, padx=18)
        self.status_lbl.pack(side=tk.LEFT, pady=10)

        tk.Label(status, text=f"{Icons.ROCKET} Parquet Field Calculator", font=FONT_SMALL,
                 bg=Colors.BG_CARD, fg=Colors.TEXT_MUTED, padx=18).pack(side=tk.RIGHT, pady=10)

    def _defaults(self):
        self.input_e.insert(0, DEFAULT_INPUT)
        self.output_e.insert(0, DEFAULT_OUTPUT)
        self._log_msg(f"{Icons.ROCKET} Application started")
        self._log_msg(f"{Icons.FOLDER} Default input: {DEFAULT_INPUT}")
        self._set_state(False)

    def _browse_in(self):
        path = filedialog.askopenfilename(filetypes=[("Parquet", "*.parquet"), ("All", "*.*")])
        if path:
            self.input_e.delete(0, tk.END)
            self.input_e.insert(0, path)

    def _browse_out(self):
        path = filedialog.askdirectory()
        if path:
            self.output_e.delete(0, tk.END)
            self.output_e.insert(0, path)

    def _analyze(self):
        path = self.input_e.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Error", "Invalid input file path")
            return
        self._set_state(True)
        self.prog_var.set(0)
        self._start_timer()
        self.worker.input_path = path
        self.worker.start_task("analyze")

    def _stop(self):
        self.worker.stop()

    def _save(self):
        out_dir = self.output_e.get().strip()
        if not out_dir:
            messagebox.showwarning("Error", "Please select output folder")
            return
        filename = os.path.basename(self.input_e.get())
        self._set_state(True)
        self._start_timer()
        self.worker.output_path = os.path.join(out_dir, f"calc_{filename}")
        self.worker.start_task("save")

    def _preview(self):
        self.worker.start_task("preview")

    def _add_field(self):
        dlg = AddFieldDialog(self.root)
        self.root.wait_window(dlg)
        if dlg.result:
            self.worker.new_field = dlg.result
            self.worker.start_task("add")

    def _manage_fields(self):
        """Open field management dialog"""
        if not self.fields:
            messagebox.showinfo("Info", "No fields to manage")
            return

        dlg = ManageFieldsDialog(self.root, self.fields)
        self.root.wait_window(dlg)

        if dlg.result:
            new_order, rename_map = dlg.result

            self._log_msg(f"{Icons.EDIT} Applying changes...")
            self._log_msg(f"   New order: {new_order[:3]}...")
            self._log_msg(f"   Renames: {rename_map}")

            # Start background task
            self._set_state(True)
            self._start_timer()
            self.worker.rename_map = rename_map if rename_map else {}
            self.worker.new_order = new_order if new_order else []
            self.worker.start_task("manage")

    def _del_fields(self):
        if not self.selected:
            messagebox.showinfo("Info", "Please select fields to delete (left-click)")
            return

        fields_list = "\n".join(sorted(self.selected))
        if messagebox.askyesno("Confirm Delete",
                               f"Delete {len(self.selected)} field(s)?\n\n{fields_list}"):
            self._set_state(True)
            self.worker.del_fields = list(self.selected)
            self.worker.start_task("delete")

    def _select_all(self):
        for name, btn in self.field_btns.items():
            if not btn.selected:
                btn.set_selected(True)
                self.selected.add(name)
        self._update_action_buttons()

    def _deselect_all(self):
        for name, btn in self.field_btns.items():
            if btn.selected:
                btn.set_selected(False)
        self.selected.clear()
        self._update_action_buttons()

    def _ins(self, text):
        self.expr_text.insert(tk.INSERT, text)
        self.expr_text.focus_set()
        self._validate()

    def _ins_func(self, func):
        pos = self.expr_text.index(tk.INSERT)
        self.expr_text.insert(tk.INSERT, func)
        self.expr_text.mark_set(tk.INSERT, f"{pos}+{len(func) - 1}c")
        self.expr_text.focus_set()
        self._validate()

    def _ins_field(self, name):
        self.expr_text.insert(tk.INSERT, f"[{name}]")
        self.expr_text.focus_set()
        self._validate()

    def _select_field(self, name, selected):
        if selected:
            self.selected.add(name)
        else:
            self.selected.discard(name)
        self._update_action_buttons()

    def _update_action_buttons(self):
        has_selection = len(self.selected) > 0
        self.del_btn.configure(state=tk.NORMAL if has_selection else tk.DISABLED)

        if self.selected:
            self.status_lbl.configure(
                text=f"{Icons.CHECK} {len(self.selected)} field(s) selected")
        else:
            self.status_lbl.configure(text=f"{Icons.CHECK} Ready")

    def _clear_expr(self):
        self.expr_text.delete('1.0', tk.END)
        self.valid_lbl.configure(text="")

    def _validate(self):
        expr = self.expr_text.get('1.0', tk.END).strip()
        lines = [l for l in expr.split('\n') if not l.strip().startswith('#')]
        expr = ' '.join(lines).strip()

        if not expr:
            self.valid_lbl.configure(text="Enter an expression", fg=Colors.TEXT_MUTED)
            self.exec_btn.configure(state=tk.DISABLED)
            return

        ok, msg = ExpressionEngine.validate(expr, [f.name for f in self.fields])
        if ok:
            self.valid_lbl.configure(text=f"{Icons.CHECK} {msg}", fg=Colors.GREEN)
            self.exec_btn.configure(state=tk.NORMAL if self.worker.df is not None else tk.DISABLED)
        else:
            self.valid_lbl.configure(text=f"{Icons.CLOSE} {msg}", fg=Colors.RED)
            self.exec_btn.configure(state=tk.DISABLED)

    def _execute(self):
        expr = self.expr_text.get('1.0', tk.END).strip()
        lines = [l for l in expr.split('\n') if not l.strip().startswith('#')]
        expr = ' '.join(lines).strip()

        name = self.result_e.get().strip()
        if not name:
            messagebox.showwarning("Error", "Please enter output field name")
            return
        if not expr:
            messagebox.showwarning("Error", "Please enter an expression")
            return

        self._set_state(True)
        self._start_timer()
        self.worker.expr = expr
        self.worker.result_name = name
        self.worker.start_task("execute")

    def _load_fields(self, fields: List[FieldInfo]):
        self.fields = fields
        self.selected.clear()
        self.field_btns.clear()

        for w in self.fields_frame.winfo_children():
            w.destroy()

        cols = 2  # Changed to 2 columns for more space per button
        for i, f in enumerate(fields):
            row, col = divmod(i, cols)
            btn = FieldButton(self.fields_frame, f, self._ins_field, self._select_field)
            btn.grid(row=row, column=col, padx=10, pady=8, sticky='nsew')
            self.field_btns[f.name] = btn

        for c in range(cols):
            self.fields_frame.columnconfigure(c, weight=1)

        self._validate()
        self._update_action_buttons()

    def _add_field_btn(self, f: FieldInfo):
        self.fields.append(f)
        self._load_fields(self.fields)

    def _remove_fields(self, names):
        self.fields = [f for f in self.fields if f.name not in names]
        self.selected -= set(names)
        self._load_fields(self.fields)

    def _handle_manage_done(self, new_order, rename_map):
        """Handle UI update after field management is complete"""
        self._log_msg(f"{Icons.SUCCESS} Updating UI...")

        # Update FieldInfo object names
        for f in self.fields:
            if f.name in rename_map:
                old_name = f.name
                new_name = rename_map[old_name]
                f.name = new_name
                self._log_msg(f"   Updated: {old_name} → {new_name}")

        # Rebuild field list in new order
        field_by_name = {f.name: f for f in self.fields}
        new_fields = []
        for name in new_order:
            if name in field_by_name:
                new_fields.append(field_by_name[name])

        self.fields = new_fields
        self.selected.clear()
        self._load_fields(self.fields)

        self._log_msg(f"{Icons.SUCCESS} UI updated with {len(self.fields)} fields")

    def _show_preview(self, df):
        self.tree.delete(*self.tree.get_children())
        cols = df.columns[:12]
        self.tree['columns'] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100, minwidth=80)
        for row in df.head(8).iter_rows():
            vals = [str(v)[:18] if v is not None else "null" for v in row[:12]]
            self.tree.insert('', 'end', values=vals)

    def _update_stats(self, s):
        self.stats = s
        self.stat_rows.configure(text=f"{s.get('rows', 0):,}")
        self.stat_fields.configure(text=f"{s.get('fields', 0)}")
        self.stat_size.configure(text=f"{s.get('size', 0):.2f} GB")
        self.stat_num.configure(text=f"{s.get('numeric', 0)}")

    def _log_msg(self, msg, tag='normal'):
        if Icons.ERROR in msg or "❌" in msg:
            tag = 'error'
        elif Icons.SUCCESS in msg or "✅" in msg:
            tag = 'success'
        elif Icons.WARNING in msg or "⚠️" in msg:
            tag = 'warning'
        elif Icons.SEARCH in msg or Icons.FOLDER in msg or Icons.INFO in msg:
            tag = 'info'
        self.log.insert(tk.END, f"{msg}\n", tag)
        if self.autoscroll.get():
            self.log.see(tk.END)

    def _clear_log(self):
        self.log.delete(1.0, tk.END)

    def _set_state(self, processing):
        self.stop_btn.set_enabled(processing)

        has_data = self.worker.df is not None

        if not processing:
            self.add_btn.configure(state=tk.NORMAL if has_data else tk.DISABLED)
            self.manage_btn.configure(state=tk.NORMAL if has_data else tk.DISABLED)
            self.sel_all_btn.configure(state=tk.NORMAL if has_data else tk.DISABLED)
            self.desel_btn.configure(state=tk.NORMAL if has_data else tk.DISABLED)
            self.save_btn.set_enabled(has_data)
            self.preview_btn.set_enabled(has_data)
            self._update_action_buttons()
            self._validate()
        else:
            self.add_btn.configure(state=tk.DISABLED)
            self.manage_btn.configure(state=tk.DISABLED)
            self.del_btn.configure(state=tk.DISABLED)
            self.sel_all_btn.configure(state=tk.DISABLED)
            self.desel_btn.configure(state=tk.DISABLED)
            self.save_btn.set_enabled(False)
            self.preview_btn.set_enabled(False)
            self.exec_btn.configure(state=tk.DISABLED)

    def _start_timer(self):
        self.t0 = time.time()
        self.timer_on = True
        self._tick()

    def _stop_timer(self):
        self.timer_on = False

    def _tick(self):
        if self.timer_on and self.t0:
            elapsed = time.time() - self.t0
            h, r = divmod(int(elapsed), 3600)
            m, s = divmod(r, 60)
            self.time_lbl.configure(text=f"{Icons.CLOCK} {h:02d}:{m:02d}:{s:02d}")
            self.root.after(1000, self._tick)

    def _poll(self):
        try:
            while True:
                msg_type, data = self.q.get_nowait()

                if msg_type == 'log':
                    self._log_msg(data)
                elif msg_type == 'progress':
                    self.prog_var.set(data)
                    self.prog_lbl.configure(text=f"{data}%")
                elif msg_type == 'fields':
                    self._load_fields(data)
                elif msg_type == 'stats':
                    self._update_stats(data)
                elif msg_type == 'new_field':
                    self._add_field_btn(data)
                elif msg_type == 'deleted':
                    self._remove_fields(data)
                elif msg_type == 'manage_done':
                    new_order, rename_map = data
                    self._handle_manage_done(new_order, rename_map)
                elif msg_type == 'preview':
                    self._show_preview(data)
                elif msg_type == 'done':
                    success, msg = data
                    self._stop_timer()
                    self._set_state(False)
                    self.status_lbl.configure(text=f"{Icons.CHECK if success else Icons.ERROR} {msg}")
                    if not success:
                        messagebox.showerror("Error", msg)
        except queue.Empty:
            pass
        self.root.after(50, self._poll)


def main():
    root = tk.Tk()

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()