import os
import shutil
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

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
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
    from pyspark.ml.feature import Imputer, PCA, StandardScaler, VectorAssembler
    from pyspark.ml.functions import vector_to_array
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

def _get_spark(app_name="breastcancer-tabular"):
    """Return the active :class:`SparkSession`, creating one if needed."""
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

def create_scree_plot(explained_variance, save_path='data/scree_plot.png'):
    """Create and save a scree plot of explained variance.

    ``explained_variance`` is the array of per-component variance ratios
    (``PCAModel.explainedVariance`` as a numpy array).
    """
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
    print("\nPCA Analysis:")
    print(f"Number of components: {k}")
    print("\nExplained variance ratio:")
    for i, ratio in enumerate(explained_variance):
        print(f"Component {i+1}: {ratio:.4f}")

    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)
    print("\nCumulative explained variance:")
    for i, var in enumerate(cumulative_variance):
        print(f"Components 1-{i+1}: {var:.4f}")

    # Analyze feature contributions
    feature_contributions, top_features = analyze_feature_contributions(pca_model, feature_names)

    # Print top contributing features for each component
    print("\nTop contributing features for each principal component:")
    for pc, features in top_features.items():
        print(f"\n{pc}:")
        for feature, contribution in features.items():
            print(f"  {feature}: {contribution:.4f}")

    # Create scree plot
    create_scree_plot(explained_variance)

    return transformed_df, pca_model, feature_contributions, top_features

def save_transformed_data(transformed_df, pca_model, feature_contributions):
    """Save the transformed data and PCA information.

    Expands the ``pca_features`` vector column into ``PC1``..``PCk`` columns,
    keeps ``Diagnosis``, and returns the resulting Spark DataFrame (which feeds
    the analysis step). CSVs are written for downstream/manual inspection.
    """
    os.makedirs('data', exist_ok=True)
    k = pca_model.getK()

    # Explode the PCA vector into one column per component.
    arr = vector_to_array(F.col("pca_features"))
    pc_cols = [arr.getItem(i).alias(f"PC{i+1}") for i in range(k)]
    transformed_df = transformed_df.select(*pc_cols, F.col("Diagnosis"))

    # The Wisconsin set is tiny (569 rows); collect to a single CSV for parity
    # with the previous output rather than a Spark part-file directory.
    transformed_df.toPandas().to_csv('data/transformed_data.csv', index=False)

    # Save PCA information
    explained_variance = pca_model.explainedVariance.toArray()
    pca_info = pd.DataFrame({
        'component': range(1, k + 1),
        'explained_variance': explained_variance,
        'cumulative_variance': np.cumsum(explained_variance)
    })
    pca_info.to_csv('data/pca_info.csv', index=False)

    # Save feature contributions
    feature_contributions.to_csv('data/feature_contributions.csv')

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

    print("\nData transformation complete!")
    print(f"Original number of features: {n_original}")
    print(f"Number of features after PCA: {pca_model.getK()}")
    print("\nTransformed data saved to 'data/transformed_data.csv'")
    print("PCA information saved to 'data/pca_info.csv'")
    print("Feature contributions saved to 'data/feature_contributions.csv'")
    print("Scree plot saved to 'data/scree_plot.png'")

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
                      case_id=None):
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
    """
    os.makedirs(output_dir, exist_ok=True)

    offset = (0, 0, 0)
    if crop:
        volume, mask, offset = crop_to_roi(volume, mask)

    volume = volume.astype(dtype)
    mask = mask.astype(np.uint8)

    out_path = os.path.join(output_dir, f"{patient_id}.npz")
    np.savez_compressed(
        out_path,
        volume=volume,
        mask=mask,
        crop_offset=np.asarray(offset, dtype=np.int32),
        case_id=np.asarray(str(case_id) if case_id is not None else str(patient_id)),
    )
    return out_path


def delete_dicom_source(dicom_dir, npz_path):
    """Delete the raw DICOM folder `dicom_dir` after preprocessing.

    Destructive — this permanently removes the original series. As a safety net the
    deletion is skipped (with a warning) unless `npz_path` exists and is non-empty,
    so a failed or partial save never costs you the source data.
    Returns True if the folder was removed.
    """
    if not npz_path or not os.path.exists(npz_path) or os.path.getsize(npz_path) == 0:
        logging.warning(
            f"Skipping deletion of {dicom_dir}: preprocessed file "
            f"{npz_path} is missing or empty."
        )
        return False
    try:
        shutil.rmtree(dicom_dir)
        logging.info(f"🗑️  Removed raw DICOM source {dicom_dir} (kept {npz_path}).")
        return True
    except OSError as e:
        logging.error(f"Failed to remove {dicom_dir}: {e}")
        return False


def dbt_series_view(ds):
    """Return the BCS-DBT view key (e.g. ``'lmlo'``) for a DBT DICOM dataset.

    Laterality lives either in the top-level ``ImageLaterality``/``Laterality`` tag
    or, for multi-frame DBT, in
    ``SharedFunctionalGroupsSequence -> FrameAnatomySequence -> FrameLaterality``.
    Combined with ``ViewPosition`` (CC/MLO) this yields the ``l/r`` + ``cc/mlo`` key
    used in the annotation boxes CSV.
    """
    lat = getattr(ds, "ImageLaterality", "") or getattr(ds, "Laterality", "")
    if not lat:
        sfg = getattr(ds, "SharedFunctionalGroupsSequence", None)
        if sfg:
            fas = getattr(sfg[0], "FrameAnatomySequence", None)
            if fas:
                lat = getattr(fas[0], "FrameLaterality", "")
    view = getattr(ds, "ViewPosition", "")
    return f"{lat}{view}".lower()


def _boxes_for_series(ds, boxes_df):
    """Rows of ``boxes_df`` matching this DICOM's PatientID and view."""
    patient = getattr(ds, "PatientID", None)
    view = dbt_series_view(ds)
    if patient is None or not view:
        return boxes_df.iloc[0:0]
    return boxes_df[(boxes_df["PatientID"] == patient)
                    & (boxes_df["View"].str.lower() == view)]


def _read_boxes(boxes_csv):
    """Read one CSV path, or a list/tuple of paths, into a single boxes DataFrame.

    Pooling several CSVs (e.g. BCS-DBT ``boxes-train`` + ``boxes-validation``, which
    hold disjoint patients under the same schema) grows the annotated set in one call.
    """
    paths = [boxes_csv] if isinstance(boxes_csv, (str, os.PathLike)) else list(boxes_csv)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def preprocess_dbt_with_boxes(root_dir="tciaDownload",
                              boxes_csv="tciaDownload/BCS-DBT-boxes-train.csv",
                              output_dir="preprocessed_data",
                              skip_empty=True,
                              slice_margin=2):
    """Preprocess Breast-Cancer-Screening-DBT series using the bounding-box annotations.

    DBT scans ship without DICOM SEG; lesions are given as 2D boxes in a separate
    CSV (``PatientID``, ``View``, ``Slice``, ``X``, ``Y``, ``Width``, ``Height``).
    For each series folder under ``root_dir`` this reads the multi-frame image,
    normalises it (see :func:`normalize_intensity`), and builds a binary lesion mask
    from the matching box rows by reusing :func:`create_mask`. Volumes are cropped to
    the lesion ROI and written as compressed ``.npz`` via :func:`save_preprocessed`.

    ``slice_margin`` extends each annotated box by this many slices on either side of
    the labelled ``Slice`` (clamped to the volume). The BCS-DBT annotation only marks
    one central slice per lesion, but the lesion itself typically spans several
    neighbouring slices in the tomosynthesis stack — without this, positive (lesion)
    slices are extremely rare, starving the 2D per-slice training loop of examples.
    Pass 0 to keep the original single-slice behaviour.

    ``boxes_csv`` may be a single path or a list of paths; pass both the train and
    validation boxes CSVs to build masks for the pooled annotated set. Series with no
    matching box are skipped when ``skip_empty`` is True (an empty mask is useless for
    the localisation model and would store the full frame).
    """
    os.makedirs(output_dir, exist_ok=True)
    boxes = _read_boxes(boxes_csv)

    saved, skipped = 0, 0
    for name in sorted(os.listdir(root_dir)):
        folder = os.path.join(root_dir, name)
        if not os.path.isdir(folder):
            continue
        dcm_files = [f for f in os.listdir(folder) if f.lower().endswith(".dcm")]
        if not dcm_files:
            continue

        ds = pydicom.dcmread(os.path.join(folder, dcm_files[0]))
        volume = ds.pixel_array.astype(np.float32)
        if volume.ndim == 2:
            volume = volume[None]  # (1, rows, cols)

        rows = _boxes_for_series(ds, boxes)
        mask = np.zeros(volume.shape, dtype=np.uint8)
        for _, r in rows.iterrows():
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
            skipped += 1
            continue

        volume = normalize_intensity(volume)
        save_preprocessed(name, volume, mask, output_dir,
                          case_id=getattr(ds, "PatientID", name))
        saved += 1
        logging.info(f"[DBT] {name}: {len(rows)} box(es), "
                     f"{int(mask.sum())} lesion voxels -> saved.")

    logging.info(f"[DBT] Saved {saved} series, skipped {skipped} without annotations.")
    return saved, skipped


def extract_patient_ids(root_dir="tciaDownload"):
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
                    print(f"Error reading {filepath}: {e}")
    return patient_ids


def process_all_mri_data(root_dir="tciaDownload", output_dir="preprocessed_data", delete_source=False):
    """
    Loop through all MRI data to preprocess them.

    Set `delete_source=True` to remove each raw DICOM folder after it has been
    successfully preprocessed into a compressed .npz (reclaims most of the disk
    space). It is off by default because it is destructive.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    
    # Store all subdirectories in a list to loop through
    mri_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    processed_count = 0
    failed_count = 0
    
    for mri_dir in mri_dirs:
        try:
            logging.info(f"Processing MRI directory: {mri_dir}")
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
            logging.error(f"Failed to process {mri_dir}: {str(e)}")
            failed_count += 1
            continue
    
    logging.info(f"Successfully processed: {processed_count}")
    logging.info(f"Failed to process: {failed_count}")
    return processed_count, failed_count

def preprocess_mri_data(series_instance_uid, local_dicom_path, output_dir="preprocessed_data", delete_source=False):
    """
    Preprocess MRI data to add its segmentations.

    If `delete_source` is True, the raw DICOM folder is deleted once the compressed
    .npz has been written successfully (see `delete_dicom_source`).
    """
    try:
        # STEP 1 - Load the MRI series
        logging.info(f"\n Loading MRI series...")
        dicom_files = [f for f in os.listdir(local_dicom_path) if f.endswith('.dcm')]
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {local_dicom_path}")
            
        # STEP 2 -Read DICOM images with single or multiple files
        if len(dicom_files) == 1:
            logging.info("Single DICOM file.")
            image = sitk.ReadImage(os.path.join(local_dicom_path, dicom_files[0]))
        else:
            logging.info("Multiple DICOM files.")
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
                    logging.info(f"Processing SEG file: {seg_file}")
                    seg_image = sitk.ReadImage(seg_path)
                    # Convert to numpy array first, then to float32
                    seg_array = sitk.GetArrayFromImage(seg_image)
                    seg_array = seg_array.astype(np.float32)
                    seg_image = sitk.GetImageFromArray(seg_array)
                    resampled_seg = resampler.Execute(seg_image)
                    seg_array = sitk.GetArrayFromImage(resampled_seg)
                    mask = np.logical_or(mask, seg_array > 0).astype(np.uint8)
                    
                elif ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':  # RTSTRUCT
                    logging.info(f"Processing RTSTRUCT file: {seg_file}")
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
                logging.warning(f"Could not process segmentation file {seg_file}: {str(e)}")
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
        logging.info(f"✅ Successfully processed {patient_id}")
        return normalized_array, mask
        
    except Exception as e:
        logging.error(f"❌ Failed to process series {series_instance_uid}: {str(e)}")
        return None, None

# Example usage
if __name__ == "__main__":
    # delete_source=True also removes each raw DICOM folder after it is safely
    # preprocessed. It is destructive, so keep it False until you have verified the
    # compressed .npz outputs are correct.
    processed_count, failed_count = process_all_mri_data(
        root_dir="tciaDownload",
        output_dir="preprocessed_data",
        delete_source=False
    )