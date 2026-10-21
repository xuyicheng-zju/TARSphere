# Build base image
FROM nvcr.io/nvidia/pytorch:25.03-py3 AS base

# tmux is for debugging, osmesa is for rendering. Put all apt-get installs in this line
RUN apt update && apt-get install -y tmux libosmesa6-dev

RUN pip install --upgrade pip

# Pin setuptools to match pyproject.toml (required by torch==2.1.0 cpp_extension for pkg_resources)
RUN pip install "setuptools>=45,<78"

# Install dependencies
RUN pip install h5py hydra-core matplotlib scikit-learn scipy tensorboard trimesh
RUN pip install numpy==1.26.4 tqdm pyyaml pytest

# Install scene_synthesizer
RUN pip install scene-synthesizer[recommend]

RUN pip install imageio python-fcl

# Install pyrender
RUN pip install pyrender && pip install PyOpenGL==3.1.5

# Diffusion dependencies. Pin transformers to a version compatible with
# diffusers==0.11.1 + huggingface-hub==0.25.2 (matches host uv resolution).
# Newer transformers (>=5) drag huggingface-hub past 1.x, which breaks the
# `from huggingface_hub import HfFolder, cached_download` imports diffusers
# 0.11.1 still uses.
RUN pip install diffusers==0.11.1 timm==1.0.15 huggingface-hub==0.25.2
RUN pip install transformers==4.48.3 tensordict

# URDF parsing (both yourdfpy and urdfpy are used by graspgenx)
RUN pip install urdfpy yourdfpy==0.0.56

# Visualizer for demo scripts (demo_object_pc / demo_scene_pc / demo_object_mesh)
RUN pip install viser

# PointTransformerV3 / sparse-conv dependencies and training utilities
RUN pip install addict yapf==0.40.1 tensorboardx sharedarray torch-geometric

# For dataset management and manifold
RUN pip install webdataset
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
RUN pip install objaverse==0.1.7
RUN mkdir -p /install/
RUN cd /install/ && git clone --recursive -j8 https://github.com/hjwdzh/Manifold.git
RUN mkdir -p /install/Manifold/build
RUN cd /install/Manifold/build
RUN cd /install/Manifold/build && cmake .. -DCMAKE_BUILD_TYPE=Release
RUN cd /install/Manifold/build && make
ENV PATH "${PATH}:/install/Manifold/build/"

# Final defensive re-pin: any earlier RUN that pulled a transitive newer
# huggingface-hub / diffusers gets undone here so diffusers==0.11.1 keeps
# working.
RUN pip install --no-deps huggingface-hub==0.25.2 diffusers==0.11.1

WORKDIR /code/