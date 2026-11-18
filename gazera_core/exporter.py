"""Export analysis results as Excel (multi-sheet), CSV and JSON.

Excel is the format most researchers hand to co-authors; JSON is what the local
API and the web workbench speak.  Both are produced from the same tidy tables so
the numbers always agree.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

__all__ = ["export_excel", "export_csv", "export_json", "to_jsonable", "safe_sheet_name"]


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas objects into JSON-safe Python values."""
    if isinstance(value, pd.DataFrame):
        return [to_jsonable(rec) for rec in value.replace({np.nan: None}).to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return to_jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def safe_sheet_name(name: str) -> str:
    """Excel sheet names: <= 31 chars, no []:*?/\\ characters."""
    for ch in "[]:*?/\\":
        name = name.replace(ch, "-")
    return name.strip()[:31] or "Sheet"


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c)[:120] for c in out.columns]
    return out


def export_excel(
    path: str | Path,
    sheets: Mapping[str, pd.DataFrame],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a multi-sheet workbook with a light GAZERA header style."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)

    header_fill = PatternFill("solid", fgColor="141413")
    header_font = Font(color="FAF9F5", bold=True, size=10)
    accent = Font(color="D97757", bold=True, size=11)

    if metadata:
        ws = wb.create_sheet("README")
        ws["A1"] = "GAZERA — analysis export"
        ws["A1"].font = Font(bold=True, size=14, color="141413")
        ws["A2"] = "Clean-room MIT research software · FEI · HEL Lab, Gyeongsang National University"
        ws["A2"].font = Font(size=9, italic=True, color="6B6A66")
        ws["A3"] = "Not affiliated with, endorsed by, or sponsored by Tobii AB."
        ws["A3"].font = Font(size=9, italic=True, color="6B6A66")
        row = 5
        for key, value in metadata.items():
            ws.cell(row=row, column=1, value=str(key)).font = accent
            cell = ws.cell(row=row, column=2, value=to_jsonable(value) if not isinstance(value, (dict, list)) else json.dumps(to_jsonable(value), ensure_ascii=False))
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            row += 1
        ws.column_dimensions["A"].width = 26
        ws.column_dimensions["B"].width = 90

    for name, df in sheets.items():
        if df is None:
            continue
        frame = _prepare(df)
        ws = wb.create_sheet(safe_sheet_name(name))
        ws.append(list(frame.columns))
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(vertical="center")
        for record in frame.itertuples(index=False):
            ws.append([None if (isinstance(v, float) and math.isnan(v)) else v for v in record])
        ws.freeze_panes = "A2"
        for idx, col in enumerate(frame.columns, start=1):
            width = max(10, min(42, max(len(str(col)) + 2, *(len(str(v)) for v in frame[col].head(200).tolist())) if len(frame) else len(str(col)) + 2))
            if any(isinstance(v, float) for v in frame[col].head(50)):
                width = max(width, 12)
            ws.column_dimensions[get_column_letter(idx)].width = width

    wb.save(path)
    return path


def export_csv(
    directory: str | Path,
    sheets: Mapping[str, pd.DataFrame],
    *,
    prefix: str = "gazera",
) -> dict[str, Path]:
    """Write one UTF-8 (BOM) CSV per table so Excel opens them correctly."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, df in sheets.items():
        if df is None:
            continue
        filename = f"{prefix}_{safe_sheet_name(name).replace(' ', '_').lower()}.csv"
        target = directory / filename
        df.to_csv(target, index=False, encoding="utf-8-sig")
        written[name] = target
    return written


def export_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
