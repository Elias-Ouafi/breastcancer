import logging
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import pydicom

import config
import lineage
import validation
from logging_setup import setup_logging

log = logging.getLogger(__name__)

# Optional heavy dependencies. Each is only needed by a specific pipeline (tabular
# PCA, MRI SEG resampling, 3D viewing...). They are imported lazily so the module —
# and the lightweight DBT box-annotation preprocessing — works without the full
# stack installed. Functions that use a missing dependency will fail only when called.
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    fetch_ucirepo = None
try:
    # Tabular pipeline now runs on Spark MLlib instead of scikit-learn.
    from pyspark.ml.feature import PCA, Imputer, StandardScaler, VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
except ImportError:
    SparkDataFrame = SparkSession = None
    Imputer = PCA = StandardScaler = VectorAssembler = None
    vector_to_array = None
    F = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import seaborn as sns
except ImportError:
    sns = None
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None
try:
    from tcia_utils import nbia
except ImportError:
    nbia = None
try:
    import itk
    import itkwidgets
    from itkwidgets import view
except ImportError:
    itk = itkwidgets = view = None

def _ensure_hadoop_home():
    """Point Hadoop at the bundled winutils on Windows, when nothing else has.

    Spark's Hadoop layer refuses to start on Windows without ``HADOOP_HOME`` and a
    ``winutils.exe`` under it -- it dies in ``SparkSubmit`` before any of our code
    runs, with a ``FileNotFoundException`` that names the variable but not what to
    put in it. The binaries are already bundled at ``<venv>/hadoop/bin``; nothing
    was pointing at them, so the tabular pipeline only ran for someone who had set
    the variable by hand in their own shell and forgotten they had.

    Set before the session is built, because pyspark launches the JVM as a
    subprocess that inherits this environment. An existing ``HADOOP_HOME`` wins: a
    machine with a real Hadoop install should keep using it.
    """
    if os.name != "nt" or os.environ.get("HADOOP_HOME"):
        return
    bundled = os.path.join(sys.prefix, "hadoop")
    if not os.path.isfile(os.path.join(bundled, "bin", "winutils.exe")):
        return
    os.environ["HADOOP_HOME"] = bundled
    os.environ["PATH"] = os.path.join(bundled, "bin") + os.pathsep + os.environ.get("PATH", "")
    log.debug("HADOOP_HOME set to bundled %s", bundled)


def _get_spark(app_name="breastcancer-tabular"):
    """Return the active :class:`SparkSession`, creating one if needed."""
    _ensure_hadoop_home()
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        # v2 skips the task-path listing that trips Hadoop's Windows NativeIO shim
        # (UnsatisfiedLinkError on access0) when winutils.exe lacks a matching hadoop.dll.
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .getOrCreate()
    )


def _pandas_to_spark(spark, pdf):
    """Load a pandas DataFrame into Spark via a temporary CSV (JVM-native read).

    ``spark.createDataFrame(pandas)`` builds a Python-backed RDD, so every
    downstream read spins up a Python worker subprocess. Writing the (small)
    tabular frame to CSV and reading it back with ``spark.read`` keeps the whole
    pipeline on the JVM side, which is more robust across environments and
    avoids a Python-worker dependency the rest of the tabular pipeline doesn't
    need. Feature columns come back as ``double`` and ``Diagnosis`` as string.
    """
    tmpdir = tempfile.mkdtemp(prefix="bc_ingest_")
    csv_path = os.path.join(tmpdir, "data.csv")
    pdf.to_csv(csv_path, index=False)
    return spark.read.csv(csv_path, header=True, inferSchema=True)


def clean_data(df, feature_cols):
    """Impute missing values and standardize features with Spark MLlib.

    ``df`` is a Spark DataFrame holding ``feature_cols`` (numeric) plus the
    ``Diagnosis`` label. Returns a DataFrame with an added ``scaled_features``
    vector column (mean-centred, unit-variance), keeping ``Diagnosis``.
    """
    # Mean-impute any missing values (parity with the old X.fillna(X.mean())).
    imputer = Imputer(inputCols=feature_cols, outputCols=feature_cols, strategy="mean")
    df = imputer.fit(df).transform(df)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled")
    assembled = assembler.transform(df)

    scaler = StandardScaler(
        inputCol="assembled", outputCol="scaled_features",
        withMean=True, withStd=True,
    )
    return scaler.fit(assembled).transform(assembled)

def analyze_feature_contributions(pca_model, feature_names):
    """Analyze and return feature contributions to principal components.

    MLlib's ``PCAModel.pc`` is a ``(n_features x k)`` DenseMatrix of loadings;
    we transpose it to ``(k x n_features)`` so each row is a principal component.
    """
    # (n_features, k) -> (k, n_features), absolute loadings.
    components = np.abs(pca_model.pc.toArray().T)
    k = components.shape[0]

    feature_contributions = pd.DataFrame(
        components,
        columns=feature_names,
        index=[f'PC{i+1}' for i in range(k)]
    )

    # For each PC, get the top 3 contributing features
    top_features = {}
    for pc in feature_contributions.index:
        top_features[pc] = feature_contributions.loc[pc].nlargest(3).to_dict()

    return feature_contributions, top_features

def create_scree_plot(explained_variance, save_path=None):
    """Create and save a scree plot of explained variance.

    ``explained_variance`` is the array of per-component variance ratios
    (``PCAModel.explainedVariance`` as a numpy array). It is a figure, so it lands
    in ``plots/`` rather than beside the data it describes.
    """
    save_path = save_path or config.SCREE_PLOT_PNG
    n_components = len(explained_variance)
    plt.figure(figsize=(10, 6))

    # Plot individual explained variance
    plt.bar(range(1, n_components + 1), explained_variance,
            alpha=0.5, align='center', label='Individual explained variance')

    # Plot cumulative explained variance
    plt.step(range(1, n_components + 1), np.cumsum(explained_variance),
             where='mid', label='Cumulative explained variance')

    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('Scree Plot')
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def apply_pca(scaled_df, feature_names, variance=0.95):
    """Apply PCA and analyze component importance.

    Spark MLlib's PCA takes a fixed number of components ``k`` rather than a
    variance target, so we first fit the full-rank PCA to read the explained
    variance, pick the smallest ``k`` that reaches ``variance`` cumulative, then
    refit with that ``k``. Returns
    ``(transformed_df, pca_model, feature_contributions, top_features)``.
    """
    n_features = len(feature_names)

    # Full-rank fit just to inspect the explained-variance profile.
    full = PCA(k=n_features, inputCol="scaled_features",
               outputCol="pca_features").fit(scaled_df)
    full_ev = full.explainedVariance.toArray()
    cumulative = np.cumsum(full_ev)
    k = int(np.searchsorted(cumulative, variance) + 1)
    k = min(k, n_features)

    # Refit keeping only the components needed to retain `variance`.
    pca_model = PCA(k=k, inputCol="scaled_features",
                    outputCol="pca_features").fit(scaled_df)
    transformed_df = pca_model.transform(scaled_df)
    explained_variance = pca_model.explainedVariance.toArray()

    # Analyze PCA results
    log.info("PCA Analysis:")
    log.info(f"Number of components: {k}")
    log.info("Explained variance ratio:")
    for i, ratio in enumerate(explained_variance):
        log.info(f"Component {i+1}: {ratio:.4f}")

    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)
    log.info("Cumulative explained variance:")
    for i, var in enumerate(cumulative_variance):
        log.info(f"Components 1-{i+1}: {var:.4f}")

    # Analyze feature contributions
    feature_contributions, top_features = analyze_feature_contributions(pca_model, feature_names)

    # Print top contributing features for each component
    log.info("Top contributing features for each principal component:")
    for pc, features in top_features.items():
        log.info(f"{pc}:")
        for feature, contribution in features.items():
            log.info(f"  {feature}: {contribution:.4f}")

    # Create scree plot
    create_scree_plot(explained_variance)

    return transformed_df, pca_model, feature_contributions, top_features

def save_transformed_data(transformed_df, pca_model, feature_contributions):
    """Save the transformed data and PCA information.

    Expands the ``pca_features`` vector column into ``PC1``..``PCk`` columns,
    keeps ``Diagnosis``, and returns the resulting Spark DataFrame (which feeds
    the analysis step). CSVs are written for downstream/manual inspection.
    """
    config.ensure_dirs(config.WISCONSIN_PREPROCESSED_DIR, config.REPORTS_DIR)
    k = pca_model.getK()

    # Explode the PCA vector into one column per component.
    arr = vector_to_array(F.col("pca_features"))
    pc_cols = [arr.getItem(i).alias(f"PC{i+1}") for i in range(k)]
    transformed_df = transformed_df.select(*pc_cols, F.col("Diagnosis"))

    # The Wisconsin set is tiny (569 rows); collect to a single CSV for parity
    # with the previous output rather than a Spark part-file directory.
    transformed_df.toPandas().to_csv(
        os.path.join(config.WISCONSIN_PREPROCESSED_DIR, 'transformed_data.csv'), index=False)

    # Save PCA information
    explained_variance = pca_model.explainedVariance.toArray()
    pca_info = pd.DataFrame({
        'component': range(1, k + 1),
        'explained_variance': explained_variance,
        'cumulative_variance': np.cumsum(explained_variance)
    })
    pca_info.to_csv(config.PCA_INFO_CSV, index=False)

    # Save feature contributions
    feature_contributions.to_csv(config.FEATURE_CONTRIBUTIONS_CSV)

    return transformed_df

def transform_data(data):
    """Transform the data through cleaning and PCA (Spark MLlib).

    ``data`` may be a pandas DataFrame (as returned by the extraction step) or an
    existing Spark DataFrame; either way it must contain the feature columns plus
    a ``Diagnosis`` column. Returns
    ``(transformed_df, pca_model, feature_contributions, top_features)`` where
    ``transformed_df`` is a Spark DataFrame of ``PC1``..``PCk`` + ``Diagnosis``.
    """
    spark = _get_spark()

    # Accept a pandas frame from ExtractData and lift it into Spark via a
    # JVM-native CSV read (see _pandas_to_spark) rather than createDataFrame.
    if SparkDataFrame is not None and not isinstance(data, SparkDataFrame):
        data = _pandas_to_spark(spark, data)

    feature_names = [c for c in data.columns if c != 'Diagnosis']
    n_original = len(feature_names)

    # Clean and preprocess data
    scaled_df = clean_data(data, feature_names)

    # Apply PCA
    transformed_df, pca_model, feature_contributions, top_features = apply_pca(
        scaled_df, feature_names
    )

    # Save transformed data
    transformed_df = save_transformed_data(transformed_df, pca_model, feature_contributions)

    log.info("Data transformation complete!")
    log.info(f"Original number of features: {n_original}")
    log.info(f"Number of features after PCA: {pca_model.getK()}")
    log.info("Transformed data saved to 'data/transformed_data.csv'")
    log.info("PCA information saved to 'data/pca_info.csv'")
    log.info("Feature contributions saved to 'data/feature_contributions.csv'")
    log.info("Scree plot saved to 'data/scree_plot.png'")

    return transformed_df, pca_model, feature_contributions, top_features


def load_dicom_volume(dicom_dir):
    """
    Load a 3D volume from a folder of DICOM slices.
    Sorts slices by InstanceNumber.
    """
    dicom_files = sorted(
        [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')],
        key=lambda f: int(pydicom.dcmread(os.path.join(dicom_dir, f)).InstanceNumber)
    )
    slices = [pydicom.dcmread(os.path.join(dicom_dir, f)).pixel_array for f in dicom_files]
    volume = np.stack(slices, axis=0)  # Shape: (depth, height, width)
    return volume

def normalize_intensity(volume, low_pct=1.0, high_pct=99.0):
    """Clip to the [low_pct, high_pct] intensity percentiles, then z-normalise.

    Plain global z-normalisation (subtract mean, divide by std over the whole
    volume) is skewed in mammography/DBT/MRI by the large air/background region:
    the mean and std mostly describe background, not tissue, so the useful
    intensity range gets compressed. Clipping outliers (background floor, any
    saturated pixels) first anchors the standardisation to the tissue range.
    """
    lo, hi = np.percentile(volume, [low_pct, high_pct])
    clipped = np.clip(volume, lo, hi)
    return (clipped - clipped.mean()) / (clipped.std() + 1e-8)


def create_mask(volume_shape, bbox):
    """
    Create a binary mask from bounding box coordinates.
    bbox keys: Start Slice, End Slice, Start Row, End Row, Start Column, End Column
    """
    mask = np.zeros(volume_shape, dtype=np.uint8)
    z0, z1 = bbox['Start Slice'], bbox['End Slice']
    y0, y1 = bbox['Start Row'], bbox['End Row']
    x0, x1 = bbox['Start Column'], bbox['End Column']
    mask[z0:z1, y0:y1, x0:x1] = 1
    return mask


def crop_to_roi(volume, mask, margin=16):
    """Crop `volume` and `mask` to the mask's bounding box plus a voxel `margin`.

    Segmentation masks are almost entirely background, so storing the full volume
    wastes space. When the mask is empty we cannot infer a region of interest, so
    the arrays are returned unchanged.
    Returns (cropped_volume, cropped_mask, offset) where `offset` is the (z, y, x)
    index of the crop origin in the original volume, so the crop can be located
    back in the full image later.
    """
    if not mask.any():
        return volume, mask, (0, 0, 0)

    nonzero = np.argwhere(mask)
    start = np.maximum(nonzero.min(axis=0) - margin, 0)
    end = np.minimum(nonzero.max(axis=0) + margin + 1, mask.shape)

    slices = tuple(slice(int(s), int(e)) for s, e in zip(start, end))
    return volume[slices], mask[slices], tuple(int(s) for s in start)


def save_preprocessed(patient_id, volume, mask, output_dir, dtype=np.float16, crop=True,
                      case_id=None, summary=None, lesion_class=None):
    """Save a preprocessed volume + mask as a single compressed .npz file.

    Three levers keep the files small:
      - `crop`: keep only the region of interest around the segmentation.
      - `dtype`: store intensities as float16 (half the size of float32); the
        precision loss is negligible for z-normalised MRI data.
      - `np.savez_compressed`: zlib-compresses the arrays; the mostly-empty mask
        shrinks by orders of magnitude.

    `case_id` is the real patient identifier used to group files for a leakage-free
    train/val/test split (several series/views can belong to one patient). It is
    stored inside the .npz; when omitted the filename (`patient_id`) is used.

    Every write goes through `validation.validate_volume_and_mask` first: this is the
    one place all four preprocessing paths funnel through, so it is the only place a
    check has to be written to cover them all. A broken volume raises here rather than
    surfacing hours later as a NaN loss. Pass `summary` a dict to collect the
    per-case stats for the lineage manifest.

    `lesion_class` ("benign" or "cancer", see `validation.LESION_CLASSES`) is written
    beside the arrays as `lesion_class` plus a 0/1 `label`. The mask cannot carry it:
    a benign lesion and a cancer paint the same pixels, so a corpus whose files only
    hold a mask has no way to answer the exam-level question. Omit it when the source
    does not say (the DCE-MRI collection has no such column) -- the keys are then
    absent, and absent means unknown rather than benign.
    """
    os.makedirs(output_dir, exist_ok=True)

    offset = (0, 0, 0)
    if crop:
        volume, mask, offset = crop_to_roi(volume, mask)

    volume = volume.astype(dtype)
    mask = mask.astype(np.uint8)

    # Validate after the cast: float16 overflow is one of the failures being caught,
    # and it does not exist until the cast has happened.
    key = str(case_id) if case_id is not None else str(patient_id)
    warnings = validation.validate_volume_and_mask(
        volume, mask, case_id=key, expect_full_frame=not crop, lesion_class=lesion_class)
    # The summary is keyed by output file, not by case: a DBT patient contributes up
    # to four views, and keying by `case_id` made each series overwrite the previous
    # one -- a manifest claiming 72 cases for 147 files, and losing 3 of its 5
    # validation warnings with them. The patient grouping stays readable as `case_id`
    # inside each entry, which is what a leakage-free split needs.
    if summary is not None:
        entry = validation.summarise(volume, mask, lesion_class=lesion_class)
        entry["case_id"] = key
        entry["warnings"] = warnings
        summary[str(patient_id)] = entry

    arrays = {
        "volume": volume,
        "mask": mask,
        "crop_offset": np.asarray(offset, dtype=np.int32),
        "case_id": np.asarray(key),
    }
    if lesion_class is not None:
        canonical, label = validation.lesion_class_label(lesion_class)
        arrays["lesion_class"] = np.asarray(canonical)
        arrays["label"] = np.asarray(label, dtype=np.uint8)

    out_path = os.path.join(output_dir, f"{patient_id}.npz")
    np.savez_compressed(out_path, **arrays)
    return out_path


def delete_dicom_source(dicom_dir, npz_path):
    """Delete the raw DICOM folder `dicom_dir` after preprocessing.

    Destructive — this permanently removes the original series. As a safety net the
    deletion is skipped (with a warning) unless `npz_path` exists and is non-empty,
    so a failed or partial save never costs you the source data.
    Returns True if the folder was removed.
    """
    if not npz_path or not os.path.exists(npz_path) or os.path.getsize(npz_path) == 0:
        log.warning(
            f"Skipping deletion of {dicom_dir}: preprocessed file "
            f"{npz_path} is missing or empty."
        )
        return False
    try:
        shutil.rmtree(dicom_dir)
        log.info(f"🗑️  Removed raw DICOM source {dicom_dir} (kept {npz_path}).")
        return True
    except OSError as e:
        log.error(f"Failed to remove {dicom_dir}: {e}")
        return False


def dbt_view_position(ds):
    """``'cc'`` or ``'mlo'`` from the header -- the half of the view key that holds.

    ``ViewPosition`` is reliable in this collection; the laterality tags beside it are
    not (see :func:`image_laterality`), so they are read from the pixels instead.
    """
    return str(getattr(ds, "ViewPosition", "")).lower()


def image_laterality(frame):
    """``'R'`` or ``'L'`` for one DBT frame, decided by which edge carries signal.

    A mammogram has the chest wall on one side and air on the other, so the edge
    sums separate the two. This is how the dataset's own reader decides laterality
    (``mazurowski-lab/duke-dbt-data``, ``duke_dbt_data.py``), whose helper for the
    DICOM tag is labelled "Unreliable - DICOM laterality is incorrect for some cases".

    Measured here, that understates it: the tag reads ``L`` on **all 262** downloaded
    series, while the pixels give 134 R and 128 L. Trusting the tag matched 147 of 253
    annotated series and pointed 23 masks at background instead of tissue -- the box
    region averaged 86 with a peak-to-peak of 119 as painted, against 399 and 378
    after the correction (the 124 correctly matched series read 416/467 as painted).
    """
    frame = np.asarray(frame)
    return "R" if frame[:, 0].sum() < frame[:, -1].sum() else "L"


# BCS-DBT ships a per-file inventory -- ``BCS-DBT-file-paths-*.csv`` -- saying which
# patient, study and view every series folder holds. Matching a box to a series is
# therefore a join on those three columns, not an inference from the pixels: the
# collection states it. Measured on the 262 downloaded series, the inference this
# replaces was right 237 times, found 253 of 260 annotated series, and took the box of
# the other acquisition of the same view 4 times (plan.md, sections 4.4 and 2026-09-13).
FILE_PATHS_COLUMNS = ("PatientID", "StudyUID", "View", "classic_path")
_VIEW_REPEAT_DIGITS = "0123456789"


def series_uid_from_classic_path(classic_path):
    """The series folder name held in a BCS-DBT ``classic_path``.

    ``classic_path`` reads ``collection/patient/study_uid/series_uid/1-1.dcm``, and
    ``series_uid`` is the folder name the downloader writes under ``TCIA_DIR`` -- which
    is what ties a row of the inventory to a folder on disk. Checked against the 262
    downloaded series: all 262 found, every ``PatientID`` agreeing.
    """
    parts = [part for part in str(classic_path).replace("\\", "/").split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"classic_path {classic_path!r} names no series folder")
    return parts[-2]


def view_position_of(view):
    """``'mlo'`` from ``'lmlo1'``: the incidence alone, no laterality, no repeat index.

    A BCS-DBT view key is laterality + incidence + an optional index for a repeated
    acquisition (``rmlo1``, ``lcc2``). The incidence is the half the DICOM header also
    carries, so it is the half that can be cross-checked.
    """
    return str(view)[1:].rstrip(_VIEW_REPEAT_DIGITS).lower()


def _read_file_paths(file_paths_csv):
    """The series inventory, indexed by series folder name.

    One path or several (train + validation cover disjoint patients). The index is
    what :func:`preprocess_dbt_with_boxes` looks a folder up by, so a duplicated
    series UID is refused rather than silently resolved to whichever row came first
    (measured: 20 311 rows, 20 311 distinct UIDs, so this is a guard, not a filter).
    """
    frames = []
    for path in _box_paths(file_paths_csv):
        df = pd.read_csv(path)
        missing = [c for c in FILE_PATHS_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path}: missing column(s) {missing} -- this should be a BCS-DBT "
                f"file-paths CSV, whose columns are {list(FILE_PATHS_COLUMNS)}; got "
                f"{list(df.columns)}")
        df = df.loc[:, list(FILE_PATHS_COLUMNS)].copy()
        df["series_uid"] = df["classic_path"].map(series_uid_from_classic_path)
        df["View"] = df["View"].astype(str).str.strip().str.lower()
        frames.append(df)
    paths = pd.concat(frames, ignore_index=True)
    duplicated = paths["series_uid"].duplicated()
    if duplicated.any():
        raise ValueError(
            f"{int(duplicated.sum())} series folder(s) listed twice across "
            f"{len(_box_paths(file_paths_csv))} file-paths CSV(s), e.g. "
            f"{paths.loc[duplicated, 'series_uid'].iloc[0]}")
    return paths.set_index("series_uid")


# The per-view status table, ``BCS-DBT-labels-*.csv``: one row per view, one flag set.
# This is the only place the collection says an exam is *normal* -- "absent from the
# boxes CSV" says "no annotated box", which is not the same thing -- so it is what an
# exam-level cancer / no-cancer target has to be built from. Measured over the three
# splits: 4 581 normal patients, 278 actionable, 112 benign, 89 cancer (5 060 total).
DBT_LABEL_COLUMNS = ("Normal", "Actionable", "Benign", "Cancer")
# Worst first. A patient is read at its worst view: one cancer view makes a cancer exam,
# the same rule the box path applies to a mixed series.
DBT_STATUS_ORDER = ("cancer", "benign", "actionable", "normal")


def read_dbt_labels(labels_csv):
    """The per-view status rows, one CSV path or several, as one DataFrame.

    The four flag columns are required: a table missing one would silently read as
    "no patient has that status", which is exactly the kind of absence that looks like
    a measurement.
    """
    frames = []
    for path in _box_paths(labels_csv):
        df = pd.read_csv(path)
        missing = [c for c in ("PatientID",) + DBT_LABEL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path}: missing column(s) {missing}; a BCS-DBT labels CSV carries "
                f"PatientID plus {list(DBT_LABEL_COLUMNS)}, got {list(df.columns)}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def dbt_patient_status(labels_csv):
    """``{PatientID: status}`` over the labels CSV(s), status being the patient's worst.

    A patient has up to four views per study and may have several studies; the status
    returned is the worst one found anywhere, per :data:`DBT_STATUS_ORDER`. A row with
    no flag set is reported as ``normal`` only if the ``Normal`` column says so -- an
    all-zero row is left out rather than assumed benign in either direction.
    """
    labels = read_dbt_labels(labels_csv)
    status = {}
    for row in labels.itertuples():
        flags = {name.lower() for name in DBT_LABEL_COLUMNS
                 if int(getattr(row, name, 0) or 0) == 1}
        if not flags:
            continue
        worst = next(name for name in DBT_STATUS_ORDER if name in flags)
        current = status.get(row.PatientID)
        if current is None or DBT_STATUS_ORDER.index(worst) < DBT_STATUS_ORDER.index(current):
            status[row.PatientID] = worst
    return status


# The column that says what an annotated box actually is. It shipped with the
# collection from the start and nothing read it, so every box -- benign or cancer --
# was painted into the same mask (plan.md, "Lire la colonne Class").
BOX_CLASS_COLUMN = "Class"

# What identifies the series a box belongs to. The inventory gives these three for
# every series folder, which is what turns the match into a join (see `_read_file_paths`).
BOX_JOIN_COLUMNS = ("PatientID", "StudyUID", "View")


def _box_paths(boxes_csv):
    """Normalise ``boxes_csv`` (one path, or several) to a list of paths."""
    if isinstance(boxes_csv, (str, os.PathLike)):
        return [boxes_csv]
    return list(boxes_csv)


def _read_boxes(boxes_csv):
    """Read one CSV path, or a list/tuple of paths, into a single boxes DataFrame.

    Pooling several CSVs (e.g. BCS-DBT ``boxes-train`` + ``boxes-validation``, which
    hold disjoint patients under the same schema) grows the annotated set in one call.

    ``Class`` is required, lowercased, and checked against
    ``validation.LESION_CLASSES``. Both rejections are deliberate: a file without the
    column would have every box treated as one undistinguished lesion (the bug this
    replaces), and an unrecognised value would be quietly dropped from the mask by the
    class filter downstream. Neither failure shows up until a model has been trained
    on it.
    """
    frames = []
    for path in _box_paths(boxes_csv):
        df = pd.read_csv(path)
        if BOX_CLASS_COLUMN not in df.columns:
            raise ValueError(
                f"{path}: no {BOX_CLASS_COLUMN!r} column. Benign and cancer boxes are "
                "not the same target, so the class is required; got columns "
                f"{list(df.columns)}")
        # The three columns the series inventory joins on. Required here rather than
        # where the join happens, so a table missing one fails on the file that is
        # wrong instead of on every series in turn.
        missing_keys = [c for c in BOX_JOIN_COLUMNS if c not in df.columns]
        if missing_keys:
            raise ValueError(
                f"{path}: missing join column(s) {missing_keys}; a box is matched to a "
                f"series by {list(BOX_JOIN_COLUMNS)}, got {list(df.columns)}")
        df[BOX_CLASS_COLUMN] = df[BOX_CLASS_COLUMN].astype(str).str.strip().str.lower()
        unknown = sorted(set(df[BOX_CLASS_COLUMN]) - set(validation.LESION_CLASSES))
        if unknown:
            raise ValueError(
                f"{path}: unrecognised {BOX_CLASS_COLUMN} value(s) {unknown}; expected "
                f"{sorted(validation.LESION_CLASSES)}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def preprocess_dbt_with_boxes(root_dir=config.TCIA_DIR,
                              boxes_csv=config.DBT_BOXES_TRAIN,
                              output_dir=config.DBT_PREPROCESSED_DIR,
                              file_paths_csv=config.DBT_FILE_PATHS,
                              skip_empty=True,
                              slice_margin=2,
                              crop=True,
                              mask_classes=("benign", "cancer")):
    """Preprocess Breast-Cancer-Screening-DBT series using the bounding-box annotations.

    DBT scans ship without DICOM SEG; lesions are given as 2D boxes in a separate
    CSV (``PatientID``, ``View``, ``Slice``, ``X``, ``Y``, ``Width``, ``Height``,
    ``Class``). For each series folder under ``root_dir`` this reads the multi-frame
    image, normalises it (see :func:`normalize_intensity`), and builds a binary lesion
    mask from the matching box rows by reusing :func:`create_mask`. Volumes are
    written as compressed ``.npz`` via :func:`save_preprocessed`.

    **The match is a join, not an inference.** ``file_paths_csv`` points at the
    collection's own inventory (``BCS-DBT-file-paths-*.csv``), which gives
    ``(PatientID, StudyUID, View)`` for every series folder: the box rows for a series
    are the rows carrying those three values. Two earlier versions inferred the view
    instead, and both were measured wrong -- the DICOM laterality tag reads ``L`` on
    all 262 downloaded series (147 of 253 series matched, 23 masks on background), and
    deriving laterality from the pixels got 237 of 262 right, reached 253 of the 260
    annotated series, and took the box of the *other* acquisition of a repeated view
    4 times. The pixels keep one job, the one the dataset's own reader gives them:
    deciding whether the stored study is rotated relative to the frame its boxes live
    in, and flipping it if so (:func:`image_laterality`). A series absent from the
    inventory is skipped and counted, because nothing can be said about which box is
    its own.

    **The class is read, and it is stored.** ``Class`` says whether a box is a
    ``benign`` finding or a ``cancer``; for a long time nothing read it, so both were
    painted into one mask and a "positive" volume meant "some lesion" -- on the 72
    patients preprocessed that way, 48 were benign and 24 cancers, so two thirds of
    the positives were not cancers. The class now travels with each file, as
    ``lesion_class`` and a 0/1 ``label`` (see :func:`save_preprocessed`), which is what
    an exam-level cancer / no-cancer decision needs. In this collection a patient's
    boxes are all of one class (verified: 82 benign-only and 59 cancer-only patients
    across the pooled train + validation CSVs, no patient mixing the two), so one
    label per file is not a simplification here; should a future CSV mix them, any
    cancer box makes the exam a cancer exam and the mixture is logged.

    ``mask_classes`` selects which boxes are *painted* into the mask, without changing
    what the label says. The default paints both: a benign lesion is still a lesion to
    localise, and dropping the benign patients would cost two thirds of the annotated
    corpus. Pass ``("cancer",)`` for a cancer-only segmentation target -- series left
    with an empty mask are then skipped like unannotated ones, and counted separately.

    ``crop`` (default True) keeps the historical behaviour of cropping to the lesion
    ROI, which keeps files small but only teaches the model to localise within an
    already-zoomed-in view. Pass ``crop=False`` to train on full, un-cropped frames
    instead — required if the model needs to run on a raw uploaded scan (see
    ``inference.predict_dbt``/``load_dbt_dicom``), since a model trained only on
    tight crops has never seen the zoomed-out scale/context of a full frame and
    over-predicts lesion almost everywhere on one (observed: ~70% of pixels flagged
    positive on a full-frame test upload).

    ``slice_margin`` extends each annotated box by this many slices on either side of
    the labelled ``Slice`` (clamped to the volume). The BCS-DBT annotation only marks
    one central slice per lesion, but the lesion itself typically spans several
    neighbouring slices in the tomosynthesis stack — without this, positive (lesion)
    slices are extremely rare, starving the 2D per-slice training loop of examples.
    Pass 0 to keep the original single-slice behaviour.

    ``boxes_csv`` and ``file_paths_csv`` may each be a single path or a list of paths;
    pass both the train and validation files to build masks for the pooled annotated
    set (disjoint patients, same schema). Series with no matching box are skipped when
    ``skip_empty`` is True (an empty mask is useless for the localisation model and
    would store the full frame). Note what such a skip does
    *not* mean: a series absent from the boxes CSV is a series with no annotated box,
    which is not the same as a normal exam -- BCS-DBT states that in a separate
    per-study labels file, which this function never reads.

    Returns ``(saved, skipped)``; the per-class breakdown goes to the log and to the
    lineage manifest written beside the volumes.
    """
    os.makedirs(output_dir, exist_ok=True)
    boxes = _read_boxes(boxes_csv)
    boxes["View"] = boxes["View"].astype(str).str.strip().str.lower()
    file_paths = _read_file_paths(file_paths_csv)
    # One group per (patient, study, view): the key the inventory gives a folder.
    boxes_by_series = dict(tuple(boxes.groupby(["PatientID", "StudyUID", "View"])))

    mask_classes = tuple(str(c).strip().lower() for c in mask_classes)
    unknown = sorted(set(mask_classes) - set(validation.LESION_CLASSES))
    if unknown:
        raise ValueError(f"mask_classes holds unrecognised class(es) {unknown}; expected "
                         f"{sorted(validation.LESION_CLASSES)}")

    saved, skipped, summary = 0, 0, {}
    saved_by_class = {name: 0 for name in validation.LESION_CLASSES}
    skipped_unannotated, skipped_unpainted, mirrored_count = 0, 0, 0
    skipped_unlisted = 0
    for name in sorted(os.listdir(root_dir)):
        folder = os.path.join(root_dir, name)
        if not os.path.isdir(folder):
            continue
        dcm_files = [f for f in os.listdir(folder) if f.lower().endswith(".dcm")]
        if not dcm_files:
            continue

        # The inventory identifies the folder, so whether it has any annotated box is
        # known before anything is decoded. That order matters: the pixels cost 7-17 s
        # and ~100 MB per series on the real collection.
        path = os.path.join(folder, dcm_files[0])
        if name not in file_paths.index:
            log.warning(f"[DBT] {name}: not in the file-paths inventory; skipping, "
                        "since which box is its own cannot be established.")
            skipped += 1
            skipped_unlisted += 1
            continue
        listed = file_paths.loc[name]
        patient, study_uid, view = listed["PatientID"], listed["StudyUID"], listed["View"]

        rows = boxes_by_series.get((patient, study_uid, view))
        if skip_empty and (rows is None or rows.empty):
            skipped += 1
            skipped_unannotated += 1
            continue
        if rows is None:
            rows = boxes.iloc[0:0]

        # Read once the series is known to be worth it, and cross-check what the header
        # does carry: PatientID, and the incidence (``ViewPosition`` is reliable here,
        # the laterality tags beside it are not). A disagreement means the folder-to-row
        # mapping is off, which no downstream count would reveal.
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        header_patient = getattr(ds, "PatientID", None)
        if header_patient is not None and str(header_patient) != str(patient):
            log.warning(f"[DBT] {name}: inventory says patient {patient}, the DICOM says "
                        f"{header_patient}; trusting the inventory.")
        header_position = dbt_view_position(ds)
        if header_position and header_position != view_position_of(view):
            log.warning(f"[DBT] {name}: inventory says view {view!r}, the DICOM says "
                        f"position {header_position!r}.")

        volume = pydicom.dcmread(path).pixel_array.astype(np.float32)
        if volume.ndim == 2:
            volume = volume[None]  # (1, rows, cols)

        stored_laterality = image_laterality(volume[0])
        mirrored = stored_laterality != view[0].upper()
        if mirrored:
            # The boxes live in the frame where the image laterality matches the view's;
            # this study is stored rotated relative to it. The volume is flipped rather
            # than the coordinates, so everything after this point -- mask, crop offset,
            # what lands in the .npz -- is in that one frame, the annotation's.
            volume = np.flip(volume, axis=(-1, -2))
            log.info(f"[DBT] {name}: inventory says view {view!r}, the pixels carry the "
                     f"breast on the {stored_laterality!r} side; reading it as a mirrored "
                     "study and flipping it (180 deg, as the reference reader does).")

        present = tuple(sorted(set(rows[BOX_CLASS_COLUMN]))) if len(rows) else ()
        # Any cancer box makes the exam a cancer exam: at the level the decision is
        # taken, one missed cancer is not offset by a correctly called benign.
        lesion_class = "cancer" if "cancer" in present else (present[0] if present else None)
        if len(present) > 1:
            log.warning(f"[DBT] {name}: boxes of several classes {present} on one series; "
                        f"labelling the exam {lesion_class!r}.")

        painted = rows[rows[BOX_CLASS_COLUMN].isin(mask_classes)]
        if skip_empty and painted.empty:
            skipped += 1
            skipped_unpainted += 1
            continue

        mask = np.zeros(volume.shape, dtype=np.uint8)
        for _, r in painted.iterrows():
            z = int(r["Slice"])
            z0 = max(0, z - slice_margin)
            z1 = min(volume.shape[0], z + 1 + slice_margin)
            bbox = {
                "Start Slice": z0, "End Slice": z1,
                "Start Row": int(r["Y"]), "End Row": int(r["Y"]) + int(r["Height"]),
                "Start Column": int(r["X"]), "End Column": int(r["X"]) + int(r["Width"]),
            }
            mask = np.logical_or(mask, create_mask(volume.shape, bbox)).astype(np.uint8)

        if skip_empty and mask.sum() == 0:
            # Boxes matched and were of a painted class, yet nothing was painted:
            # a degenerate box (zero width/height, or off the frame).
            log.warning(f"[DBT] {name}: {len(painted)} box(es) painted no voxel; skipping.")
            skipped += 1
            skipped_unpainted += 1
            continue

        volume = normalize_intensity(volume)
        save_preprocessed(name, volume, mask, output_dir, crop=crop,
                          case_id=patient if patient is not None else name,
                          lesion_class=lesion_class, summary=summary)
        # Provenance of the geometry, not a property of the volume: the manifest is
        # where a reader asks how a case was built.
        if name in summary:
            summary[name]["mirrored"] = bool(mirrored)
            summary[name]["view"] = view
            summary[name]["study_uid"] = study_uid
        saved += 1
        mirrored_count += int(mirrored)
        if lesion_class is not None:
            saved_by_class[lesion_class] += 1
        log.info(f"[DBT] {name}: {len(painted)}/{len(rows)} box(es) painted, "
                 f"class {lesion_class or 'unknown'}, "
                 f"{int(mask.sum())} lesion voxels -> saved.")

    # Written last, and only on a completed pass: a folder with no manifest is a
    # folder whose run was interrupted (same rule as the DCE-MRI path).
    lineage.write_manifest(
        output_dir,
        source=root_dir,
        parameters={
            "pipeline": "preprocess_dbt_with_boxes",
            "boxes": [lineage.relative_path(p) for p in _box_paths(boxes_csv)],
            "file_paths": [lineage.relative_path(p) for p in _box_paths(file_paths_csv)],
            "crop": crop,
            "slice_margin": slice_margin,
            "skip_empty": skip_empty,
            "mask_classes": list(mask_classes),
            "saved_by_class": saved_by_class,
            "mirrored_series": mirrored_count,
            "skipped_unannotated": skipped_unannotated,
            "skipped_unpainted": skipped_unpainted,
            "skipped_unlisted": skipped_unlisted,
        },
        cases=summary,
        warnings=[w for case in summary.values() for w in case.get("warnings", [])],
    )

    log.info(f"[DBT] Saved {saved} series "
             f"({saved_by_class['cancer']} cancer, {saved_by_class['benign']} benign, "
             f"{mirrored_count} read as mirrored), "
             f"skipped {skipped} ({skipped_unannotated} without any box, "
             f"{skipped_unpainted} annotated but empty for classes {list(mask_classes)}, "
             f"{skipped_unlisted} absent from the file-paths inventory).")
    return saved, skipped


def extract_patient_ids(root_dir=config.TCIA_DIR):
    """Walk `root_dir` and return the set of PatientIDs found in the DICOM files."""
    patient_ids = set()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.dcm'):
                filepath = os.path.join(dirpath, filename)
                try:
                    ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                    patient_ids.add(ds.PatientID)
                except Exception as e:
                    log.info(f"Error reading {filepath}: {e}")
    return patient_ids


def process_all_mri_data(root_dir=config.TCIA_DIR, output_dir=config.DBT_PREPROCESSED_DIR,
                         delete_source=False):
    """
    Loop through all MRI data to preprocess them.

    Set `delete_source=True` to remove each raw DICOM folder after it has been
    successfully preprocessed into a compressed .npz (reclaims most of the disk
    space). It is off by default because it is destructive.
    """
    # No logging.basicConfig here: this is a library function, and configuring the
    # root logger from one would hijack logging for every caller. Entry points call
    # logging_setup.setup_logging() instead.
    os.makedirs(output_dir, exist_ok=True)

    # Store all subdirectories in a list to loop through
    mri_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    processed_count = 0
    failed_count = 0
    
    for mri_dir in mri_dirs:
        try:
            log.info(f"Processing MRI directory: {mri_dir}")
            local_dicom_path = os.path.join(root_dir, mri_dir)
            
            # Process the MRI data
            volume, mask = preprocess_mri_data(
                series_instance_uid=mri_dir,
                local_dicom_path=local_dicom_path,
                output_dir=output_dir,
                delete_source=delete_source
            )
            
            if volume is not None and mask is not None:
                processed_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            log.error(f"Failed to process {mri_dir}: {str(e)}")
            failed_count += 1
            continue
    
    log.info(f"Successfully processed: {processed_count}")
    log.info(f"Failed to process: {failed_count}")
    return processed_count, failed_count

def preprocess_mri_data(series_instance_uid, local_dicom_path,
                        output_dir=config.DBT_PREPROCESSED_DIR, delete_source=False):
    """
    Preprocess MRI data to add its segmentations.

    If `delete_source` is True, the raw DICOM folder is deleted once the compressed
    .npz has been written successfully (see `delete_dicom_source`).
    """
    try:
        # STEP 1 - Load the MRI series
        log.info(" Loading MRI series...")
        dicom_files = [f for f in os.listdir(local_dicom_path) if f.endswith('.dcm')]
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {local_dicom_path}")
            
        # STEP 2 -Read DICOM images with single or multiple files
        if len(dicom_files) == 1:
            log.info("Single DICOM file.")
            image = sitk.ReadImage(os.path.join(local_dicom_path, dicom_files[0]))
        else:
            log.info("Multiple DICOM files.")
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(local_dicom_path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        
        # STEP 3 - Convert images to numpy array first, then to float32
        image_array = sitk.GetArrayFromImage(image)
        image_array = image_array.astype(np.float32)
        image = sitk.GetImageFromArray(image_array)
        # Use the right spacing and size
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        standard_spacing = (1.0, 1.0, 1.0)
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(standard_spacing)
        resampler.SetSize([int(sz * spc / nspc) for sz, spc, nspc in zip(original_size, original_spacing, standard_spacing)])
        resampler.SetInterpolator(sitk.sitkLinear)
        resampled_image = resampler.Execute(image)
        
        # STEP 4 - Convert to numpy array for processing
        image_array = sitk.GetArrayFromImage(resampled_image)
        
        # STEP 5 - Normalize the array
        normalized_array = normalize_intensity(image_array)
        mask = np.zeros_like(normalized_array, dtype=np.uint8)
        
        # STEP 6 - Segment the data
        seg_files = [f for f in os.listdir(local_dicom_path) if f.endswith('.dcm')]
        for seg_file in seg_files:
            seg_path = os.path.join(local_dicom_path, seg_file)
            try:
                # Read DICOM file
                ds = pydicom.dcmread(seg_path)
                
                # Check if it's a segmentation file
                if ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4':
                    log.info(f"Processing SEG file: {seg_file}")
                    seg_image = sitk.ReadImage(seg_path)
                    # Convert to numpy array first, then to float32
                    seg_array = sitk.GetArrayFromImage(seg_image)
                    seg_array = seg_array.astype(np.float32)
                    seg_image = sitk.GetImageFromArray(seg_array)
                    resampled_seg = resampler.Execute(seg_image)
                    seg_array = sitk.GetArrayFromImage(resampled_seg)
                    mask = np.logical_or(mask, seg_array > 0).astype(np.uint8)
                    
                elif ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':  # RTSTRUCT
                    log.info(f"Processing RTSTRUCT file: {seg_file}")
                    # Convert RTSTRUCT to binary mask
                    rtstruct = itk.imread(seg_path)
                    rtstruct_resampled = itk.resample_image_filter(
                        rtstruct,
                        size=resampled_image.GetSize(),
                        spacing=standard_spacing
                    )
                    rtstruct_array = itk.GetArrayFromImage(rtstruct_resampled)
                    mask = np.logical_or(mask, rtstruct_array > 0).astype(np.uint8)
                    
            except Exception as e:
                log.warning(f"Could not process segmentation file {seg_file}: {str(e)}")
                continue
        
        # STEP 7 - Save preprocessed data (compressed to keep files small)
        patient_id = os.path.basename(local_dicom_path)
        npz_path = save_preprocessed(patient_id, normalized_array, mask, output_dir)
        if mask.any():
            view(normalized_array, mask, ui_collapsed=True)

        # STEP 8 - Optionally reclaim disk space by deleting the raw DICOM source
        if delete_source:
            delete_dicom_source(local_dicom_path, npz_path)

        # The end
        log.info(f"✅ Successfully processed {patient_id}")
        return normalized_array, mask
        
    except Exception as e:
        log.error(f"❌ Failed to process series {series_instance_uid}: {str(e)}")
        return None, None

# --------------------------------------------------------------------------- #
# DCE-MRI (Duke-Breast-Cancer-MRI): multiphase subtraction + box annotations
# --------------------------------------------------------------------------- #

# Duke-Breast-Cancer-MRI mixes two SeriesDescription conventions across its cohort
# (protocol/scanner varied over the years the collection was acquired):
#   A) "ax dyn pre", "ax dyn 1st pass" .. "ax dyn 4th pass"       (spelled-out pass)
#   B) "ax 3d dyn" (bare = pre-contrast), "Ph1/ax 3d dyn" .. "Ph4/..." (Ph-prefixed)
# Verified on a 10-patient sample: exactly 5/10 use each convention, so both must be
# handled or half the cohort silently loses its DCE series.
_DCE_PASS_WORDS = [("1st", 1), ("2nd", 2), ("3rd", 3), ("4th", 4)]
_DCE_PH_PREFIX_RE = re.compile(r"\bph\s*(\d)\b")


def _dce_phase_rank(series_description):
    """Return the phase rank (0=pre-contrast, 1..4=post-contrast passes) for a
    Duke-Breast-Cancer-MRI ``SeriesDescription``, or None if it doesn't match the
    dynamic-contrast protocol (e.g. the unrelated "ax t1 tse +c" or SEG series)."""
    desc = (series_description or "").lower()
    m = _DCE_PH_PREFIX_RE.search(desc)
    if m:
        return int(m.group(1))
    for key, rank in _DCE_PASS_WORDS:
        if key in desc:
            return rank
    if "pre" in desc:
        return 0
    if "dyn" in desc:
        # Convention B's pre-contrast series carries no "Ph"/"pass"/"pre" marker at
        # all -- a bare dynamic series is the baseline by elimination.
        return 0
    return None


def group_dce_series_by_patient(root_dir):
    """Walk ``root_dir``'s series subfolders (one ``SeriesInstanceUID`` per folder,
    as written by :func:`ExtractData.download_dce_mri_series`) and group them by
    ``PatientID -> {phase_rank: folder_path}``.

    Only folders whose ``SeriesDescription`` matches the DCE dynamic protocol are
    kept; unrelated series (T1 TSE, SEG) are ignored here.
    """
    groups = {}
    for name in sorted(os.listdir(root_dir)):
        folder = os.path.join(root_dir, name)
        if not os.path.isdir(folder):
            continue
        dcm_files = [f for f in os.listdir(folder) if f.lower().endswith(".dcm")]
        if not dcm_files:
            continue
        ds = pydicom.dcmread(os.path.join(folder, dcm_files[0]), stop_before_pixels=True)
        rank = _dce_phase_rank(getattr(ds, "SeriesDescription", ""))
        if rank is None:
            continue
        pid = getattr(ds, "PatientID", None)
        if pid is None:
            continue
        groups.setdefault(pid, {})[rank] = folder
    return groups


def _load_series_volume(dicom_dir):
    """Load one DICOM series folder into a ``(depth, H, W)`` float32 array, sorted
    by ``InstanceNumber`` (slice order within the series)."""
    dcm_files = sorted(
        (f for f in os.listdir(dicom_dir) if f.lower().endswith(".dcm")),
        key=lambda f: int(pydicom.dcmread(os.path.join(dicom_dir, f),
                                          stop_before_pixels=True).InstanceNumber),
    )
    slices = [pydicom.dcmread(os.path.join(dicom_dir, f)).pixel_array for f in dcm_files]
    return np.stack(slices, axis=0).astype(np.float32)


def _read_mri_boxes(boxes_path):
    """Read the Duke-Breast-Cancer-MRI ``Annotation_Boxes`` file (``.xlsx`` or the
    ``.csv`` produced by :func:`ExtractData.clean_mri_annotation`) into a DataFrame.

    Expected columns (1-indexed, inclusive, as published by TCIA): ``Patient ID``,
    ``Start Row``, ``End Row``, ``Start Column``, ``End Column``, ``Start Slice``,
    ``End Slice``. Column names are normalised (stripped) so minor header variants
    still match.
    """
    if str(boxes_path).lower().endswith(".xlsx"):
        df = pd.read_excel(boxes_path)
    else:
        df = pd.read_csv(boxes_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def preprocess_dce_mri_with_boxes(root_dir, boxes_path,
                                  output_dir=config.DCE_MRI_PREPROCESSED_DIR,
                                  post_phase_rank=2, crop=True):
    """Preprocess Duke-Breast-Cancer-MRI series into subtraction volumes + box masks.

    For each patient with both a pre-contrast (rank 0) and the chosen post-contrast
    pass (``post_phase_rank``) series available, this:

    ``post_phase_rank`` defaults to 2 (the *second* post-contrast pass). Zhou et al.,
    "U-Net breast lesion segmentations for breast dynamic contrast-enhanced MRI"
    (https://pmc.ncbi.nlm.nih.gov/articles/PMC10658935/) compared subtraction inputs
    head-to-head on this exact task and found second-post-contrast subtraction
    significantly better than first (DSC p<0.05, for both 2D and 3D U-Nets) -- the
    second pass sits nearer peak enhancement, so malignant uptake stands out more
    against background parenchyma. Verified here that both passes are available for
    the same 186/189 patients, so the switch costs no sample size. Pass
    ``post_phase_rank=1`` to reproduce the earlier first-pass behaviour.

    1. Loads both phases (:func:`_load_series_volume`) -- they are acquired in the
       same session without repositioning, so no inter-phase registration is applied
       for this first pass.
    2. Computes the enhancement subtraction ``post - pre`` (clipped at 0: only
       contrast uptake is informative for lesion conspicuity), then z-normalises it
       with :func:`normalize_intensity` -- the same convention already used by the
       DBT path, so the resulting ``.npz`` is a drop-in for the existing
       ``imaging/`` training/inference code (single-channel volume + mask).
    3. Builds a binary lesion mask from the matching row(s) in the TCIA annotation
       boxes file via :func:`create_mask` (1-indexed bounds are converted to the
       0-indexed slicing ``create_mask``/numpy expect).

    Patients with mismatched phase shapes (rare acquisition inconsistencies) or no
    matching box row are skipped. Returns ``(saved, skipped)``.
    """
    os.makedirs(output_dir, exist_ok=True)
    boxes = _read_mri_boxes(boxes_path)
    groups = group_dce_series_by_patient(root_dir)

    saved, skipped, summary = 0, 0, {}
    for pid, phases in groups.items():
        if 0 not in phases or post_phase_rank not in phases:
            log.warning(f"[MRI] {pid}: missing pre or post-phase {post_phase_rank} series, skipping.")
            skipped += 1
            continue

        rows = boxes[boxes["Patient ID"] == pid]
        if rows.empty:
            skipped += 1
            continue

        pre = _load_series_volume(phases[0])
        post = _load_series_volume(phases[post_phase_rank])
        if pre.shape != post.shape:
            log.warning(f"[MRI] {pid}: phase shape mismatch {pre.shape} vs {post.shape}, skipping.")
            skipped += 1
            continue

        subtraction = np.clip(post - pre, a_min=0, a_max=None)

        mask = np.zeros(subtraction.shape, dtype=np.uint8)
        for _, r in rows.iterrows():
            bbox = {
                # TCIA boxes are 1-indexed inclusive; create_mask/numpy slicing is
                # 0-indexed exclusive on the end, so subtract 1 only from the starts.
                "Start Slice": int(r["Start Slice"]) - 1, "End Slice": int(r["End Slice"]),
                "Start Row": int(r["Start Row"]) - 1, "End Row": int(r["End Row"]),
                "Start Column": int(r["Start Column"]) - 1, "End Column": int(r["End Column"]),
            }
            mask = np.logical_or(mask, create_mask(subtraction.shape, bbox)).astype(np.uint8)

        subtraction = normalize_intensity(subtraction)
        save_preprocessed(pid, subtraction, mask, output_dir, crop=crop, case_id=pid,
                          summary=summary)
        saved += 1
        log.info(f"[MRI] {pid}: {len(rows)} box(es), {int(mask.sum())} lesion voxels -> saved.")

    # Written last, and only on a completed pass: a folder with no manifest is a
    # folder whose run was interrupted, which is exactly what you want to know.
    lineage.write_manifest(
        output_dir,
        source=root_dir,
        parameters={
            "pipeline": "preprocess_dce_mri_with_boxes",
            "post_phase_rank": post_phase_rank,
            "crop": crop,
            "boxes": lineage.relative_path(boxes_path),
            "skipped": skipped,
        },
        cases=summary,
        warnings=[w for case in summary.values() for w in case.get("warnings", [])],
    )

    log.info(f"[MRI] Saved {saved} patients, skipped {skipped}.")
    return saved, skipped


def make_demo_case(source_npz, out_path, slice_index, slim=True, slab=12):
    """Write ``source_npz`` to ``out_path`` with a ``forced_slice`` key added.

    ``inference._localize_lesion`` scores only ``forced_slice`` when present instead
    of scanning the whole volume for the highest-confidence slice. This exists
    because that scan is currently unreliable on full-frame DCE-MRI (confidence
    saturates near 1.0 on almost every slice -- verified 0/186 on held-out patients,
    see plan.md §4.1/§4.2): the model segments a lesion well *once shown the right
    slice*, it just cannot reliably find that slice on its own yet. Demo cases are
    curated by hand (pick a real, verified-good slice) so the app has something
    trustworthy to show while that ranking problem is being worked on separately --
    this is a known, documented limitation, not a hidden shortcut.

    ``slim`` (the default) keeps a small **slab** centred on that slice instead of the
    whole volume: ``slab`` slices either side, so ``2 * slab + 1`` in total. Since
    ``forced_slice`` means only the centre slice is ever scored, carrying all ~176
    costs ~30 MB per case for nothing; a slab of 25 is ~4 MB, small enough to live in
    git so the demo works straight out of a clone. The neighbours are kept (rather
    than the centre alone) so the UI can offer slice-by-slice navigation and a MIP,
    which is what makes a lesion legible -- it should appear and disappear as you
    scroll, not just sit there in a single frame.

    The dropped depth is preserved in ``source_n_slices`` and the slab's first index
    is folded into ``crop_offset[0]``, so the app still reports "slice 52 of 176" --
    the same numbers a full-volume case produces. ``slab=0`` keeps the centre slice
    alone; ``slim=False`` keeps the whole volume (e.g. to re-derive another slice).
    """
    with np.load(source_npz) as data:
        kwargs = {k: data[k] for k in data.files}

    slice_index = int(slice_index)
    if not slim:
        kwargs["forced_slice"] = np.asarray(slice_index)
        np.savez_compressed(out_path, **kwargs)
        return out_path

    depth = int(kwargs["volume"].shape[0])
    if not 0 <= slice_index < depth:
        raise IndexError(f"slice_index {slice_index} out of range for depth {depth}.")

    offset = kwargs.get("crop_offset")
    offset = np.zeros(3, dtype=np.int32) if offset is None else np.asarray(offset, dtype=np.int32)

    lo = max(0, slice_index - int(slab))
    hi = min(depth, slice_index + int(slab) + 1)
    for key in ("volume", "mask"):
        if key in kwargs:
            kwargs[key] = kwargs[key][lo:hi]

    # The slab's first slice becomes index 0, so its original position moves into the
    # z offset -- exactly the mechanism `save_preprocessed`'s crop already uses, which
    # is what makes `best_slice` and `render_overlay_png` come out unchanged.
    kwargs["crop_offset"] = np.asarray([offset[0] + lo, offset[1], offset[2]], dtype=np.int32)
    kwargs["forced_slice"] = np.asarray(slice_index - lo)
    kwargs["source_n_slices"] = np.asarray(offset[0] + depth, dtype=np.int32)
    np.savez_compressed(out_path, **kwargs)
    return out_path


# Example usage
if __name__ == "__main__":
    setup_logging(logfile="preprocess.log")
    # delete_source=True also removes each raw DICOM folder after it is safely
    # preprocessed. It is destructive, so keep it False until you have verified the
    # compressed .npz outputs are correct.
    processed_count, failed_count = process_all_mri_data(
        root_dir=config.TCIA_DIR,
        output_dir=config.DBT_PREPROCESSED_DIR,
        delete_source=False
    )