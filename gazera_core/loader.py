"""Vendor-neutral gaze data loader.

GAZERA reads only plain delimited text (CSV / TSV) that a researcher exported
from whatever eye-tracking software they use.  No vendor SDK, binary format or
proprietary library is touched: the loader sniffs encoding, delimiter and header
row, then maps the file's own column names onto GAZERA's canonical schema.

Every mapping is a suggestion.  Anything can be overridden with ``mapping=``.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CANONICAL_FIELDS",
    "recording_columns",
    "GazeData",
    "detect_mapping",
    "find_aoi_columns",
    "load_gaze",
    "load_gaze_bytes",
    "normalise_columns",
    "parse_text",
]

# ---------------------------------------------------------------------------
# Canonical schema and column patterns
# ---------------------------------------------------------------------------

#: Canonical fields understood by the rest of the pipeline.
CANONICAL_FIELDS: tuple[str, ...] = (
    "timestamp",
    "gaze_x",
    "gaze_y",
    "pupil_left",
    "pupil_right",
    "pupil",
    "validity_left",
    "validity_right",
    "validity",
    "event",
    "event_duration",
    "fixation_x",
    "fixation_y",
    "subject",
    "trial",
    "condition",
    "media_width",
    "media_height",
)

#: ``(field, [regex, ...])`` in priority order.  Patterns are matched against a
#: normalised column name: lower-case, single spaces, no surrounding blanks.
#: The list is intentionally built from *generic* export vocabulary (timestamp,
#: gaze point, pupil diameter, validity, eye movement type, AOI hit ...), which
#: is shared by many eye-tracking tools, not from any single vendor's manual.
_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "timestamp",
        (
            r"^recording timestamp$",
            r"^recording timestamp \[(ms|us|µs|s)\]$",
            r"^timestamp$",
            r"^timestamp \[(ms|us|µs|s)\]$",
            r"^device timestamp.*$",
            r"^time stamp$",
            r"^time$",
            r"^time[ _]?(ms|us|µs|s)$",
            r"^timestamp[ _]?(ms|us|µs|s)$",
            r"^time \((ms|us|µs|s)\)$",
            r"^(recording )?timestamp.*$",
        ),
    ),
    (
        "gaze_x",
        (
            r"^gaze point x$",
            r"^gaze point x \((adcs|mcs|dacs|dcs|px).*\)$",
            r"^gaze x$",
            r"^gaze position x$",
            r"^gaze_x$",
            r"^x$",
            r"^interpolated gaze point x$",
            r"^interpolated gaze x$",
            r"^.*gaze point.* x$",
        ),
    ),
    (
        "gaze_y",
        (
            r"^gaze point y$",
            r"^gaze point y \((adcs|mcs|dacs|dcs|px).*\)$",
            r"^gaze y$",
            r"^gaze position y$",
            r"^gaze_y$",
            r"^y$",
            r"^interpolated gaze point y$",
            r"^interpolated gaze y$",
            r"^.*gaze point.* y$",
        ),
    ),
    (
        "pupil_left",
        (
            r"^pupil diameter left$",
            r"^pupil diameter left \[mm\]$",
            r"^pupil left$",
            r"^left pupil diameter$",
            r"^pupil diameter \(left\)$",
        ),
    ),
    (
        "pupil_right",
        (
            r"^pupil diameter right$",
            r"^pupil diameter right \[mm\]$",
            r"^pupil right$",
            r"^right pupil diameter$",
            r"^pupil diameter \(right\)$",
        ),
    ),
    (
        "pupil",
        (
            r"^pupil diameter$",
            r"^pupil diameter \[mm\]$",
            r"^pupil$",
        ),
    ),
    (
        "validity_left",
        (
            r"^validity left$",
            r"^left validity$",
            r"^left eye validity$",
        ),
    ),
    (
        "validity_right",
        (
            r"^validity right$",
            r"^right validity$",
            r"^right eye validity$",
        ),
    ),
    (
        "validity",
        (
            r"^validity$",
            r"^gaze validity$",
            r"^sample validity$",
        ),
    ),
    (
        "event",
        (
            r"^eye movement type$",
            r"^event type$",
            r"^eye movement$",
            r"^movement type$",
            r"^gaze event type$",
        ),
    ),
    (
        "event_duration",
        (
            r"^gaze event duration$",
            r"^event duration$",
            r"^fixation duration$",
        ),
    ),
    (
        "fixation_x",
        (
            r"^fixation point x$",
            r"^fixation point x \(.*\)$",
            r"^fixation x$",
        ),
    ),
    (
        "fixation_y",
        (
            r"^fixation point y$",
            r"^fixation point y \(.*\)$",
            r"^fixation y$",
        ),
    ),
    (
        "subject",
        (
            r"^participant name$",
            r"^participant id$",
            r"^participant$",
            r"^subject id$",
            r"^subject name$",
            r"^subject$",
            r"^ppn$",
            r"^pp$",
        ),
    ),
    (
        "trial",
        (
            r"^recording name$",
            r"^recording id$",
            r"^recording$",
            r"^trial id$",
            r"^trial name$",
            r"^trial$",
        ),
    ),
    (
        "condition",
        (
            r"^presented stimulus name$",
            r"^stimulus name$",
            r"^condition$",
            r"^group$",
            r"^media name$",
        ),
    ),
    ("media_width", (r"^media width$", r"^stimulus width$", r"^screen width$")),
    ("media_height", (r"^media height$", r"^stimulus height$", r"^screen height$")),
)

_AOI_PATTERN = re.compile(r"^aoi hit \[(?P<name>.+)\]$")

#: Columns that look like a time but are not the master clock.
_TIME_BLOCKLIST = (
    "duration",
    "time to first",
    "ttff",
    "time of day",
    "elapsed",
)


def recording_columns(frame: pd.DataFrame) -> list[str]:
    """Columns that identify one recording.

    ``condition`` is part of the identity: exports often carry one row block per
    stimulus with no trial column, and grouping by subject alone would merge two
    conditions into one timeline.
    """
    return [c for c in ("subject", "trial", "condition") if c in frame and frame[c].notna().any()]


def normalise_columns(columns: Iterable[Any]) -> list[str]:
    """Lower-case, trim and collapse whitespace for matching purposes."""
    out = []
    for col in columns:
        name = str(col).replace("\ufeff", "").strip().lower()
        name = re.sub(r"\s+", " ", name)
        out.append(name)
    return out


def detect_mapping(columns: Sequence[Any]) -> dict[str, str]:
    """Suggest a canonical mapping for ``columns``.

    Returns a dict ``canonical field -> original column name``.  Only fields
    that could be matched are present.  Resolution is first-match-wins over the
    priority-ordered patterns, so a plain ``Gaze point X`` wins over
    ``Interpolated gaze point X``.
    """
    originals = list(columns)
    normalised = normalise_columns(originals)
    used: set[int] = set()
    mapping: dict[str, str] = {}

    for field_name, patterns in _PATTERNS:
        found = False
        for pattern in patterns:
            rx = re.compile(pattern)
            for idx, name in enumerate(normalised):
                if idx in used or not name:
                    continue
                if field_name == "timestamp" and any(b in name for b in _TIME_BLOCKLIST):
                    continue
                if rx.match(name):
                    mapping[field_name] = originals[idx]
                    used.add(idx)
                    found = True
                    break
            if found:
                break
    return mapping


def find_aoi_columns(columns: Sequence[Any]) -> dict[str, str]:
    """Return ``aoi name -> column`` for ``AOI hit [name]`` style columns."""
    result: dict[str, str] = {}
    for col in columns:
        name = str(col).replace("\ufeff", "").strip()
        match = _AOI_PATTERN.match(name.lower())
        if match:
            raw = name
            # recover the original case of the AOI name inside the brackets
            start, end = raw.find("["), raw.rfind("]")
            aoi = raw[start + 1 : end].strip() if start >= 0 and end > start else match.group("name")
            result[aoi] = col
    return result


# ---------------------------------------------------------------------------
# Text parsing
# ---------------------------------------------------------------------------

_ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr", "utf-16", "latin-1")
_DELIMITERS = (",", "\t", ";", "|")


def _decode(raw: bytes) -> tuple[str, str]:
    """Decode bytes with the first encoding that works cleanly."""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16"), "utf-16"
    best: tuple[float, str, str] | None = None
    for enc in _ENCODINGS:
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        if enc == "latin-1":
            score = 1.0  # last resort: always decodes, often mojibake
        else:
            score = text.count("\ufffd") / max(len(text), 1)
        if best is None or score < best[0]:
            best = (score, text, enc)
        if score == 0 and enc in ("utf-8-sig", "utf-8"):
            break
    if best is None:  # pragma: no cover - latin-1 always succeeds
        return raw.decode("utf-8", errors="replace"), "utf-8"
    return best[1], best[2]


def _count_delimiter(line: str, delim: str) -> int:
    return line.count(delim)


def _sniff_delimiter(lines: Sequence[str]) -> str:
    """Pick the delimiter that splits the sample most consistently."""
    sample = [ln for ln in lines[:200] if ln.strip()]
    if not sample:
        return ","
    best_delim, best_score = ",", -1.0
    for delim in _DELIMITERS:
        counts = [_count_delimiter(ln, delim) for ln in sample[:40]]
        if not counts or max(counts) == 0:
            continue
        best_count = max(counts)
        consistent = counts.count(best_count)
        # a real delimiter splits nearly every row the same way; require that a
        # meaningful share of rows carries the maximum field count.
        score = best_count * 2 + consistent * 3
        if consistent < max(2, len(counts) // 5):
            score *= 0.5
        if score > best_score:
            best_delim, best_score = delim, float(score)
    return best_delim


def _find_header_index(lines: Sequence[str], delimiter: str) -> int:
    """Find the header row, skipping free-text preamble (common in exports)."""
    limit = min(len(lines), 120)
    best_idx, best_fields = 0, 0
    for idx in range(limit):
        line = lines[idx]
        if not line.strip():
            continue
        fields = next(csv.reader([line], delimiter=delimiter), [])
        n_fields = len(fields)
        if n_fields < 2:
            continue
        # the header must be followed by rows with a similar field count
        nxt = [
            len(next(csv.reader([lines[j]], delimiter=delimiter), []))
            for j in range(idx + 1, min(idx + 4, len(lines)))
            if lines[j].strip()
        ]
        consistent = sum(1 for n in nxt if n == n_fields)
        score = n_fields + consistent * 4
        keyword = 1 if re.search(
            r"(timestamp|gaze|pupil|validity|participant|recording|aoi)", lines[idx], re.I
        ) else 0
        score += keyword * 6
        if score > best_fields:
            best_idx, best_fields = idx, score
    return best_idx


def parse_text(text: str, delimiter: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse delimited text into a string DataFrame plus parse metadata."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if delimiter is None:
        delimiter = _sniff_delimiter(lines)
    header_idx = _find_header_index(lines, delimiter)
    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(
        io.StringIO(body),
        sep=delimiter,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
    )
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    # drop fully empty trailing columns/rows produced by stray separators
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, [c for c in df.columns if c and not c.lower().startswith("unnamed:")] or df.columns]
    df = df[~df.apply(lambda row: all(str(v).strip() == "" for v in row), axis=1)]
    meta = {
        "delimiter": delimiter,
        "header_row_index": header_idx,
        "delimiter_label": {",": "comma", "\t": "tab", ";": "semicolon", "|": "pipe"}.get(
            delimiter, delimiter
        ),
    }
    return df.reset_index(drop=True), meta


# ---------------------------------------------------------------------------
# Canonical frame construction
# ---------------------------------------------------------------------------

def _to_number(series: pd.Series, decimal_comma: bool = False) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "-": np.nan, "N/A": np.nan})
    if decimal_comma:
        text = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        # tolerate thousands separators such as 1,234.5
        text = text.str.replace(r"(?<=\d),(?=\d{3}\b)", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def _infer_timestamp_unit(values: pd.Series, column_name: str) -> tuple[pd.Series, str]:
    """Return timestamps in milliseconds and the detected unit."""
    name = str(column_name).lower()
    numeric = values.astype(float)
    dt = numeric.diff().dropna()
    dt = dt[dt > 0]
    median_dt = float(dt.median()) if len(dt) else float("nan")
    unit = "ms"
    if re.search(r"\[(us|µs)\]", name):
        unit = "us"
    elif re.search(r"\[s\]$", name):
        unit = "s"
    elif re.search(r"\[ms\]", name):
        unit = "ms"
    elif median_dt == median_dt:  # not NaN
        if median_dt > 1000:  # e.g. 60-120 Hz sampled in microseconds
            unit = "us"
        elif median_dt < 1:  # seconds with decimals
            unit = "s"
    factor = {"ms": 1.0, "us": 0.001, "s": 1000.0}[unit]
    return numeric * factor, unit


def _resolve_validity(
    frame: pd.DataFrame, zero_is_invalid: bool | None = None
) -> tuple[pd.Series, str]:
    """Build a boolean valid mask from validity columns.

    Convention is detected, not assumed: with a 0-4 scale (the common export
    convention) ``0`` means valid, while with a plain 0/1 flag ``1`` means valid.
    Override with ``zero_is_invalid=True/False``.
    """
    cols = [c for c in ("validity_left", "validity_right", "validity") if c in frame]
    if not cols:
        return pd.Series(True, index=frame.index), "no validity column"
    values = frame[cols]
    if zero_is_invalid is None:
        vals = values.to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size and np.nanmax(finite) > 1:
            zero_is_invalid = True
        elif finite.size and np.nanmax(finite) <= 1 and np.nanmin(finite) >= 0:
            zero_is_invalid = False
        else:
            zero_is_invalid = True
    if zero_is_invalid:
        mask = (values == 0).all(axis=1)
    else:
        mask = (values > 0).all(axis=1)
    mask = mask & values.notna().all(axis=1)
    return mask.fillna(False), "zero-is-valid" if zero_is_invalid else "one-is-valid"


def _series_or_nan(frame: pd.DataFrame, field_name: str, index: pd.Index) -> pd.Series:
    if field_name in frame:
        return frame[field_name]
    return pd.Series(np.nan, index=index, dtype=float)


@dataclass
class GazeData:
    """A loaded recording set in GAZERA's canonical schema."""

    frame: pd.DataFrame
    raw: pd.DataFrame
    mapping: dict[str, str]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def aoi_columns(self) -> dict[str, str]:
        return find_aoi_columns(self.raw.columns)

    @property
    def has_gaze(self) -> bool:
        return {"gaze_x", "gaze_y"}.issubset(self.frame.columns) and self.frame["gaze_x"].notna().any()

    def subjects(self) -> list[str]:
        if "subject" not in self.frame:
            return []
        return sorted({str(s) for s in self.frame["subject"].dropna().unique()})

    def conditions(self) -> list[str]:
        if "condition" not in self.frame:
            return []
        return sorted({str(s) for s in self.frame["condition"].dropna().unique()})

    def groups(self) -> list[tuple[str | None, str | None]]:
        """Recording groups: one per (subject, trial, condition)."""
        frame = self.frame
        cols = recording_columns(frame)
        if not cols:
            return [(None, None)]
        keys = frame[cols].drop_duplicates()
        out = []
        for _, row in keys.iterrows():
            out.append((row.get("subject"), row.get("trial")))
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "rows": int(len(self.frame)),
            "columns": int(len(self.raw.columns)),
            "subjects": self.subjects(),
            "conditions": self.conditions(),
            "recordings": int(len(self.groups())),
            **self.meta,
        }


def build_frame(
    raw: pd.DataFrame,
    mapping: Mapping[str, str] | None = None,
    *,
    decimal_comma: bool = False,
    zero_is_invalid: bool | None = None,
    screen: tuple[float, float] | None = None,
) -> GazeData:
    """Turn a raw string DataFrame into the canonical GAZERA frame."""
    if mapping is None:
        mapping = detect_mapping(raw.columns)
    mapping = {k: v for k, v in mapping.items() if v in raw.columns}

    missing = [f for f in ("gaze_x", "gaze_y") if f not in mapping]
    if missing:
        raise ValueError(
            "Cannot find gaze coordinates. Map 'gaze_x' and 'gaze_y' explicitly. "
            f"Missing: {', '.join(missing)}. Columns: {list(raw.columns)[:12]}"
        )

    work = pd.DataFrame(index=raw.index)
    for field_name, column in mapping.items():
        if field_name in ("subject", "trial", "condition", "event"):
            work[field_name] = raw[column].astype(str).str.strip().replace({"": np.nan})
        elif field_name in ("media_width", "media_height"):
            values = _to_number(raw[column], decimal_comma)
            work[field_name] = values
        else:
            work[field_name] = _to_number(raw[column], decimal_comma)

    meta: dict[str, Any] = {}
    # --- time base --------------------------------------------------------
    if "timestamp" in work:
        ts, unit = _infer_timestamp_unit(work["timestamp"], mapping["timestamp"])
        work["time_ms"] = ts
        meta["timestamp_unit"] = unit
        meta["timestamp_source"] = mapping["timestamp"]
    else:
        # synthesise a time base if the export has none: 60 Hz assumption is
        # explicit in the metadata so the researcher can correct it.
        work["time_ms"] = np.arange(len(work), dtype=float) * (1000.0 / 60.0)
        meta["timestamp_unit"] = "synthetic-60hz"
    # rebase per recording so every recording starts at 0 ms
    group_cols = recording_columns(work)
    if group_cols:
        work["time_ms"] = work["time_ms"] - work.groupby(group_cols, dropna=False)["time_ms"].transform("min")
    else:
        work["time_ms"] = work["time_ms"] - float(np.nanmin(work["time_ms"])) if len(work) else work["time_ms"]

    # --- pupil ------------------------------------------------------------
    pupils = [c for c in ("pupil_left", "pupil_right") if c in work and work[c].notna().any()]
    if pupils:
        work["pupil"] = work[pupils].mean(axis=1, skipna=True)
    elif "pupil" not in work:
        work["pupil"] = np.nan

    # --- validity ---------------------------------------------------------
    valid, convention = _resolve_validity(work, zero_is_invalid=zero_is_invalid)
    work["valid"] = valid
    meta["validity_convention"] = convention

    # --- screen size ------------------------------------------------------
    width = float(screen[0]) if screen else np.nan
    height = float(screen[1]) if screen else np.nan
    if "media_width" in work and work["media_width"].notna().any():
        width = float(work["media_width"].dropna().median())
    if "media_height" in work and work["media_height"].notna().any():
        height = float(work["media_height"].dropna().median())
    work.attrs["screen"] = (width, height)

    for col in ("gaze_x", "gaze_y"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if np.isnan(width) or np.isnan(height):
        width = float(np.nanpercentile(work["gaze_x"].dropna(), 99)) if work["gaze_x"].notna().any() else np.nan
        height = float(np.nanpercentile(work["gaze_y"].dropna(), 99)) if work["gaze_y"].notna().any() else np.nan
        work.attrs["screen"] = (width, height)

    # samples without coordinates are not usable even if flagged valid
    work["valid"] = work["valid"] & work["gaze_x"].notna() & work["gaze_y"].notna()

    work = work.reset_index(drop=True)
    meta.update(
        {
            "encoding": meta.get("encoding"),
            "screen_width": None if math.isnan(width) else round(width, 3),
            "screen_height": None if math.isnan(height) else round(height, 3),
        }
    )
    return GazeData(frame=work, raw=raw.reset_index(drop=True), mapping=dict(mapping), meta=meta)


def load_gaze_bytes(
    raw: bytes,
    *,
    mapping: Mapping[str, str] | None = None,
    delimiter: str | None = None,
    decimal_comma: bool = False,
    zero_is_invalid: bool | None = None,
    screen: tuple[float, float] | None = None,
    filename: str | None = None,
) -> GazeData:
    """Load gaze data from raw bytes (CSV / TSV, any common text encoding)."""
    text, encoding = _decode(raw)
    df, parse_meta = parse_text(text, delimiter=delimiter)
    data = build_frame(
        df,
        mapping=mapping,
        decimal_comma=decimal_comma,
        zero_is_invalid=zero_is_invalid,
        screen=screen,
    )
    data.meta.update(parse_meta)
    data.meta["encoding"] = encoding
    if filename:
        data.meta["filename"] = filename
    return data


def load_gaze(
    path: str | Path,
    *,
    mapping: Mapping[str, str] | None = None,
    delimiter: str | None = None,
    decimal_comma: bool = False,
    zero_is_invalid: bool | None = None,
    screen: tuple[float, float] | None = None,
) -> GazeData:
    """Load a CSV/TSV gaze export from disk."""
    path = Path(path)
    data = load_gaze_bytes(
        path.read_bytes(),
        mapping=mapping,
        delimiter=delimiter,
        decimal_comma=decimal_comma,
        zero_is_invalid=zero_is_invalid,
        screen=screen,
        filename=path.name,
    )
    return data
