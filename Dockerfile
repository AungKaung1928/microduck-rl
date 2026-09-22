# CPU-only, reproducible. Runs the checks that need no assets by default.
#
#   docker build -t microduck-rl .
#   docker run --rm microduck-rl                       # test_ppo, test_export
#   docker run --rm microduck-rl bash -c './fetch_assets.sh && ./verify.sh'
#
# assets/ is not baked in: the 3D model files are CC BY-SA-NC and the repo
# only ships the fetch script. The second command pulls them (one shallow
# clone, ~24 MB) and runs the full suite inside the container.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MUJOCO_GL=disable

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash", "-c", "python test_ppo.py && python test_export.py"]
