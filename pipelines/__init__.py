"""Prefect flows that run the project's pipelines end to end.

Until now the DCE-MRI pipeline existed only as prose: the README told you to call
``download_dce_mri_series``, then ``preprocess_dce_mri_with_boxes``, then
``python -m imaging.train``, then ``python -m imaging.evaluate`` -- four commands with
undocumented ordering, no record of what ran, and nothing to resume from when step
three failed forty minutes in.

These flows make that sequence executable and observable. See :mod:`pipelines.dce_mri`.
"""
