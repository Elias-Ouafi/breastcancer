# The demo, reproducible on a machine that has only Docker.
#
#   docker compose up --build      then open http://127.0.0.1:5000
#
# What this image is NOT: the training environment. Training needs CUDA, ~90 GB of
# DICOM and a GPU; none of that belongs in an image whose job is to make a reviewer
# see the thing work in one command. So the install is deliberately narrow -- no
# PySpark, no JVM, no TCIA client (ITK and SimpleITK left the dependency list
# altogether on 2026-09-19). That is the difference between a ~700 MB
# image and a multi-gigabyte one, and none of it is reachable from the demo path.

FROM python:3.12-slim AS base

# libgomp1: PyTorch's CPU kernels are OpenMP-threaded and the slim image omits it,
#   which surfaces as an ImportError on `import torch` rather than as a missing lib.
# git: lineage.git_revision() shells out to it; without it a manifest silently loses
#   the one field that says which code ran.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch, from the index that serves it. The default wheel carries bundled
# CUDA libraries that cannot be used here: the container is not given a GPU. Done
# before the requirements file so the resolution below finds torch already
# satisfied and never reaches for the PyPI wheel.
RUN pip install --no-cache-dir "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu

# The demo's actual dependency surface -- Flask serves it, numpy and Pillow render
# the slices -- read from the same file a host install uses, so the image and a
# clone cannot drift apart (a test pins that they do not). Installed before the
# source is copied so editing a template does not reinstall PyTorch.
COPY requirements-demo.txt ./
RUN pip install --no-cache-dir -r requirements-demo.txt

# Source last: it changes on every commit, the layers above almost never do.
COPY config.py logging_setup.py inference.py run_demo.py validation.py lineage.py ./
COPY app/ ./app/
COPY imaging/ ./imaging/
COPY models/dce_mri_p2_negfix/ ./models/dce_mri_p2_negfix/
COPY models/sliceclf/ ./models/sliceclf/
COPY data/curated_data/demo_cases/ ./data/curated_data/demo_cases/

# Nothing here needs root. If the image is ever run with a bind mount, this is what
# keeps it from writing root-owned files into the host checkout.
RUN useradd --create-home --uid 10001 demo && chown -R demo:demo /app
USER demo

EXPOSE 5000

# Fails fast and loudly if a COPY above ever stops matching where the code looks --
# a missing checkpoint should mark the container unhealthy, not surface as a stack
# trace on the first click.
#
# --fast-check, not --check: the full preflight loads the U-Net and scores a case,
# which is the right thing to do once at startup and the wrong thing to do every
# 30 seconds inside a 10-second timeout on a CPU-only container. The container's
# own CMD runs the full check when it starts.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python run_demo.py --fast-check || exit 1

# run_demo.py preflights the checkpoint and the demo cases, then starts Flask. Inside
# a container app.server.bind_host() returns 0.0.0.0 -- the container's own isolated
# interface. Reachability from outside is decided by the port mapping in
# docker-compose.yml, which binds the host's loopback only.
CMD ["python", "run_demo.py"]
