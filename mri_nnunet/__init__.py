"""The nnU-Net MRI corpus: its own preprocessing, in its own corpus, beside the served one.

``TransformData`` and ``data/silver/dce_mri_p2/`` are what the demo's checkpoint was trained on
(one channel, the subtraction; z-scored over the whole volume). They stay exactly as they are.
This package builds, from the raw DICOM, what nnU-Net v2 needs instead: several channels in
native geometry, N4-corrected, organ-masked, registered, resampled to a spacing chosen from
the lesions, normalised inside the mask, with pseudo-masks made from the published boxes.

* ``steps``      pure numpy functions (boxes -> ellipsoids, normalisation, spacing choice)
* ``sitk_io``    SimpleITK: DICOM with geometry, N4, mask, registration, resampling
* ``pipeline``   ingest (DICOM -> native NIfTI) and process (native -> the case)
* ``spacing``    the box-size distribution and the target spacing it implies
* ``export``     the ``nnUNet_raw`` dataset
* ``qc``         the report on random cases
* ``registry``   registry, exclusions and per-case log

Parameters are in ``config.yaml``. ``python -m mri_nnunet --help``.
"""
