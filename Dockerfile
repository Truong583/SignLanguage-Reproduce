# Use --build-arg BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
# for newer GPU generations; this is a different numerical environment.
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/assets/hf-cache TORCH_HOME=/workspace/assets/torch-cache \
    MPLCONFIGDIR=/tmp/matplotlib TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HOME=/tmp
WORKDIR /opt/signlanguage
COPY requirements-repro.txt .
RUN python -m pip install --no-cache-dir --only-binary=:all: --force-reinstall --no-deps ninja==1.11.1.3 \
    && python -m pip install --no-cache-dir -r requirements-repro.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/python-environment.lock
COPY repro ./repro
COPY scripts ./scripts
COPY tests ./tests
COPY configs_repro ./configs_repro
COPY run.py ./run.py
COPY update.py diagnose.py status.py setup_machine.py service.py ./
COPY docs/AUDIT.md docs/paper_targets.json docs/PHOENIX14T_SUITE.md ./docs/
COPY MixSignGraph/utils/sacrebleu.py MixSignGraph/utils/Rouge.py MixSignGraph/utils/video_augmentation.py ./MixSignGraph/utils/
ARG RUN_TESTS=0
RUN if [ "$RUN_TESTS" = "1" ]; then python -m pytest -q tests; fi
WORKDIR /workspace
ENV PYTHONPATH=/opt/signlanguage
USER 1000:1000
CMD ["python", "-m", "repro.smoke", "--device", "cpu"]
