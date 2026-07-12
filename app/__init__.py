"""MRI cancer-detection web app (Flask).

See ``app/server.py`` for the web layer and ``app/predictor.py`` for the pluggable
prediction backend (mock by default; real U-Net wired via MRI_APP_BACKEND=unet).
"""
