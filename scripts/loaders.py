"""Dataset loading. One loader for every dataset, known or not.

   Returns a common interface: (X, y, metadata) where X is a 2D float array
   (n_samples x n_features), y is optional labels (or None), and metadata is a
   dict of extra info.

   There is deliberately no per-dataset code and no registry: any folder
   data/<name>/ is loaded the same way, covering csv/tsv/txt, npy, npz, and
   h5ad. Where a dataset's data lives is declared in an optional "## Loading"
   section of its DATA_DESCRIPTION.md (see parse_loading_spec). Nothing
   downstream of this file should ever branch on which dataset is loaded.

   This module never touches the network. Fetching a missing file is a
   separate setup step (scripts/fetch_data.py), so an analysis run only ever
   reads what is already under data/<name>/.
"""
import csv
import re
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


# ── Generic loader ──

GENERIC_EXTENSIONS = {".csv", ".tsv", ".txt", ".npy", ".npz", ".h5ad"}
# url/md5 are read by scripts/fetch_data.py, not by the loader itself.
# image_shape (e.g. "28,28,3") marks flattened features as an image; npy/npz arrays with >2 dims get it automatically.
LOADING_KEYS = {"file", "label_column", "id_column", "delimiter", "x_key", "y_key", "url", "md5", "image_shape"}
# A dense float64 matrix bigger than this is refused with a clear message
# instead of letting the process run out of memory partway through.
EXTRA_NA_TOKENS = ["?", "-", "--", ".", "na", "none", "missing"]  # feature columns only, see _read_delimited
MAX_DENSE_BYTES = 4 * 10**9


def parse_loading_spec(description_path):
    """Read the optional '## Loading' section of DATA_DESCRIPTION.md: lines of
    `key: value` (bullets and backticks allowed). Only lowercase snake_case (letters, digits, underscores)
    keys are treated as settings, so ordinary prose in that section is
    ignored, but a misspelled setting (e.g. `lable_column`) is an error
    rather than being silently dropped."""
    description_path = Path(description_path)
    if not description_path.exists():
        return {}
    text = description_path.read_text(encoding="utf-8", errors="replace")
    section = re.search(r"^##\s+Loading\s*$(.*?)(?=^##\s|\Z)", text, flags=re.M | re.S)
    if not section:
        return {}

    spec = {}
    for line in section.group(1).splitlines():
        m = re.match(r"^\s*[-*]?\s*`?([a-z][a-z0-9_]*)`?\s*:\s*(.+?)\s*$", line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip("`'\" ")
        if key not in LOADING_KEYS:
            raise ValueError(
                f"Unknown setting '{key}' in the '## Loading' section of {description_path}. "
                f"Allowed: {sorted(LOADING_KEYS)}"
            )
        spec[key] = value
    return spec


def _find_data_file(folder, spec):
    if "file" in spec:
        path = folder / spec["file"]
        if not path.is_file():
            raise ValueError(f"'file: {spec['file']}' in DATA_DESCRIPTION.md does not exist under {folder}")
        return path

    candidates = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in GENERIC_EXTENSIONS
    )
    if not candidates:
        raise ValueError(
            f"No data file found in {folder} (looked for {sorted(GENERIC_EXTENSIONS)}). "
            "Add one, or point to it with 'file: <name>' under '## Loading' in DATA_DESCRIPTION.md."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Found {len(candidates)} candidate data files in {folder}: {[p.name for p in candidates]}. "
            "Say which one to use with 'file: <name>' under '## Loading' in DATA_DESCRIPTION.md."
        )
    return candidates[0]


def _check_dense_size(n_rows, n_cols, source):
    if n_rows * n_cols * 8 > MAX_DENSE_BYTES:
        raise ValueError(
            f"{source} would need a dense {n_rows:,} x {n_cols:,} float64 matrix "
            f"(~{n_rows * n_cols * 8 / 1e9:.1f} GB), above the {MAX_DENSE_BYTES / 1e9:.0f} GB limit. "
            "This pipeline works on dense arrays; subsample or reduce features before loading."
        )


def _read_delimited(path, spec):
    import pandas as pd

    delimiter = spec.get("delimiter")
    if delimiter is None:
        if path.suffix.lower() == ".tsv":
            delimiter = "\t"
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(8192)
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
            except csv.Error:
                delimiter = ","
    elif delimiter.lower() in ("\\t", "tab"):
        delimiter = "\t"

    # Common "missing" spellings beyond pandas' defaults (which already cover "", NA, NaN, null, ...).
    # Applied to feature columns only, so a label class that happens to be spelled "none" or "-" stays a label.
    protected = {spec.get("label_column"), spec.get("id_column")}
    for encoding in ("utf-8", "latin-1"):
        try:
            header = pd.read_csv(path, sep=delimiter, nrows=0, encoding=encoding)
            na_values = {c: EXTRA_NA_TOKENS for c in header.columns if c not in protected}
            return pd.read_csv(path, sep=delimiter, encoding=encoding, na_values=na_values)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path.name} as utf-8 or latin-1")


def _load_table(path, spec):
    import pandas as pd

    df = _read_delimited(path, spec)

    label_col, id_col = spec.get("label_column"), spec.get("id_column")
    for setting, col in (("label_column", label_col), ("id_column", id_col)):
        if col and col not in df.columns:
            raise ValueError(
                f"'{setting}: {col}' in DATA_DESCRIPTION.md is not a column of {path.name}. "
                f"Columns include: {list(df.columns)[:15]}"
            )

    y = df[label_col].to_numpy() if label_col else None
    features = df.drop(columns=[c for c in (label_col, id_col) if c])

    non_numeric = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        raise ValueError(
            f"{path.name} has non-numeric feature columns {non_numeric[:10]}; "
            "this loader currently handles numeric features only. "
            "If one of them is a label, declare it as 'label_column' under '## Loading'."
        )

    _check_dense_size(features.shape[0], features.shape[1], path.name)
    metadata = {
        "source": path.stem,
        "source_file": path.name,
        "modality": "tabular",
        "feature_names": [str(c) for c in features.columns],
    }
    if label_col:
        metadata["label_column"] = label_col
    return features.to_numpy(dtype=np.float64), y, metadata


def _as_matrix(arr, source):
    arr = np.asarray(arr)
    if arr.ndim < 2:
        raise ValueError(f"{source} is a {arr.ndim}-D array; need at least 2-D (samples x features).")
    image_shape = arr.shape[1:] if arr.ndim > 2 else None
    X = arr.reshape(arr.shape[0], -1).astype(np.float64)
    return X, image_shape


def _load_npy(path, spec):
    X, image_shape = _as_matrix(np.load(path, allow_pickle=False), path.name)
    _check_dense_size(*X.shape, path.name)
    metadata = {"source": path.stem, "source_file": path.name, "modality": "array"}
    if image_shape:
        metadata["image_shape"] = image_shape
    return X, None, metadata


def _load_npz(path, spec):
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.keys())
        x_key = spec.get("x_key", "X")
        if x_key not in keys:
            raise ValueError(
                f"{path.name} has no array named '{x_key}' (found {keys}). "
                "Name the features array with 'x_key: <name>' under '## Loading'."
            )
        X, image_shape = _as_matrix(data[x_key], f"{path.name}[{x_key}]")

        y_key = spec.get("y_key")
        if y_key is None and "y" in keys:
            y_key = "y"
        y = None
        if y_key:
            if y_key not in keys:
                raise ValueError(f"'y_key: {y_key}' not found in {path.name} (found {keys}).")
            y = np.asarray(data[y_key]).reshape(len(X), -1)
            y = y[:, 0] if y.shape[1] == 1 else y.reshape(len(X), -1)
            if y.ndim != 1:
                raise ValueError(f"Labels '{y_key}' in {path.name} are not one value per sample.")

    _check_dense_size(*X.shape, path.name)
    metadata = {"source": path.stem, "source_file": path.name, "modality": "array"}
    if image_shape:
        metadata["image_shape"] = image_shape
    return X, y, metadata


def _load_h5ad(path, spec):
    import anndata as ad

    adata = ad.read_h5ad(path)
    _check_dense_size(adata.n_obs, adata.n_vars, path.name)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

    label_col = spec.get("label_column")
    y = None
    if label_col:
        if label_col not in adata.obs.columns:
            raise ValueError(
                f"'label_column: {label_col}' is not in {path.name}'s obs. "
                f"obs columns: {list(adata.obs.columns)[:15]}"
            )
        y = adata.obs[label_col].to_numpy()

    metadata = {
        "source": path.stem,
        "source_file": path.name,
        "modality": "annotated_matrix",
        "feature_names": adata.var_names.astype(str).tolist(),
    }
    return np.asarray(X, dtype=np.float64), y, metadata


def _parse_image_shape(text, n_features, source):
    """'28,28,3' or '28x28x3' -> (28, 28, 3); must multiply out to the feature count."""
    try:
        shape = tuple(int(v) for v in re.split(r"[,x\s]+", text.strip("()[] ")) if v)
    except ValueError:
        raise ValueError(f"image_shape '{text}' in DATA_DESCRIPTION.md is not a list of integers, e.g. 28,28,3")
    if not shape or int(np.prod(shape)) != n_features:
        raise ValueError(f"image_shape {shape} declared for {source} does not multiply to its {n_features} features")
    return shape


def load_generic(folder):
    """Load whatever single data file lives in data/<name>/, guided by the
    optional '## Loading' section of that folder's DATA_DESCRIPTION.md."""
    folder = Path(folder)
    spec = parse_loading_spec(folder / "DATA_DESCRIPTION.md")
    path = _find_data_file(folder, spec)
    suffix = path.suffix.lower()

    if suffix in (".csv", ".tsv", ".txt"):
        X, y, metadata = _load_table(path, spec)
    elif suffix == ".npy":
        X, y, metadata = _load_npy(path, spec)
    elif suffix == ".npz":
        X, y, metadata = _load_npz(path, spec)
    elif suffix == ".h5ad":
        X, y, metadata = _load_h5ad(path, spec)
    else:
        raise ValueError(f"Unsupported data file type: {path.name}")

    if "image_shape" in spec:
        metadata["image_shape"] = _parse_image_shape(spec["image_shape"], X.shape[1], path.name)

    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError(f"{path.name} loaded as shape {X.shape}; need at least 2 samples and 1 feature.")
    if y is not None and len(y) != X.shape[0]:
        raise ValueError(f"Labels have {len(y)} entries but there are {X.shape[0]} samples.")
    return X, y, metadata


def load_dataset(name: str):
    folder = DATA_ROOT / name
    if not folder.is_dir():
        available = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir()) if DATA_ROOT.is_dir() else []
        raise ValueError(f"Unknown dataset '{name}': there is no folder {folder}. Datasets present: {available}")
    return load_generic(folder)
