#!/usr/bin/env bash
set -euo pipefail

# Recreate the remote myconda environment used for ov2/Overcooked experiments.
# Source environment:
#   /root/miniconda3/envs/myconda on hz-4.matpool.com:26991
#   Python 3.12.4, JAX 0.4.28 CUDA 12, PyTorch 2.3.1+cu121.
#
# Usage on a fresh Linux CUDA server:
#   bash tools/install_myconda_cuda12.sh
#
# Optional:
#   ENV_NAME=ov2 bash tools/install_myconda_cuda12.sh

ENV_NAME="${ENV_NAME:-myconda}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not on PATH. Install Miniconda/Anaconda first, then rerun." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env ${ENV_NAME} already exists; using it."
else
  conda create -y -n "${ENV_NAME}" python=3.12.4 pip=24.0 setuptools=69.5.1 wheel=0.43.0
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip==24.0 setuptools==69.5.1 wheel==0.43.0

# JAX CUDA wheel. The original env used:
#   jax==0.4.28
#   jaxlib==0.4.28+cuda12.cudnn89
python -m pip install \
  --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  "jax==0.4.28" \
  "jaxlib==0.4.28+cuda12.cudnn89"

# PyTorch CUDA 12.1 wheels.
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.3.1+cu121" \
  "torchvision==0.18.1+cu121" \
  "torchaudio==2.3.1+cu121"

REQ_FILE="$(mktemp)"
cat > "${REQ_FILE}" <<'REQ'
absl-py==2.1.0
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
asttokens==2.4.1
attrs==25.3.0
blinker==1.9.0
brax==0.10.3
certifi==2025.1.31
cffi==2.0.0
charset-normalizer==3.4.1
chex==0.1.86
click==8.1.8
cloudpickle==3.0.0
colorama==0.4.6
comm==0.2.2
contourpy==1.2.1
cryptography==49.0.0
cycler==0.12.1
debugpy==1.8.2
decorator==5.1.1
distrax==0.1.5
dm-env==1.6
dm-tree==0.1.9
docker-pycreds==0.4.0
docopt-ng==0.9.0
etils==1.12.2
executing==2.0.1
Farama-Notifications==0.0.4
filelock==3.13.1
flashbax==0.1.0
Flask==3.1.0
flask-cors==5.0.1
flax==0.8.4
fonttools==4.53.0
fsspec==2024.2.0
gast==0.6.0
gitdb==4.0.12
GitPython==3.1.44
glfw==2.8.0
grpcio==1.64.1
gym==0.26.2
gym-notices==0.0.8
gymnasium==1.1.1
gymnax==0.0.8
h5py==3.11.0
humanize==4.12.2
hydra-core==1.3.2
idna==3.10
imageio==2.37.0
importlib_resources==6.5.2
ipykernel==6.29.4
ipython==8.25.0
itsdangerous==2.2.0
jaxopt==0.8.3
jedi==0.19.1
Jinja2==3.1.3
joblib==1.4.2
jsonpickle==4.1.2
jupyter_client==8.6.2
jupyter_core==5.7.2
kiwisolver==1.4.5
lightgbm==4.4.0
Markdown==3.6
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.9.0
matplotlib-inline==0.1.7
mdurl==0.1.2
ml-dtypes==0.3.2
ml_collections==1.0.0
mpmath==1.3.0
msgpack==1.1.0
mujoco==3.1.3
mujoco-mjx==3.1.3
munch==4.0.0
nest-asyncio==1.6.0
networkx==3.2.1
numpy==1.26.4
omegaconf==2.3.0
opencv-python==4.10.0.84
opt_einsum==3.4.0
optax==0.2.2
orbax-checkpoint==0.5.16
packaging==24.1
pandas==2.2.2
parso==0.8.4
pdfminer.six==20260107
pdfplumber==0.11.10
pettingzoo==1.24.3
pexpect==4.9.0
pillow==12.2.0
platformdirs==4.2.2
prompt_toolkit==3.0.47
protobuf==4.25.3
psutil==6.0.0
ptyprocess==0.7.0
pure-eval==0.2.2
py-cpuinfo==9.0.0
pycparser==3.0
pydantic==2.11.2
pydantic_core==2.33.1
Pygments==2.18.0
PyMuPDF==1.27.2.3
PyOpenGL==3.1.9
pyparsing==3.1.2
pypdf==6.13.2
pypdfium2==5.10.1
python-dateutil==2.9.0.post0
pytinyrenderer==0.0.14
pytz==2024.1
PyYAML==6.0.2
pyzmq==26.0.3
reportlab==4.5.1
requests==2.32.3
rich==14.0.0
sacred==0.8.7
safetensors==0.5.3
scikit-learn==1.5.0
scipy==1.12.0
seaborn==0.13.2
sentry-sdk==2.25.1
setproctitle==1.3.5
simplejson==3.20.1
six==1.16.0
smmap==5.0.2
spyder-kernels==2.5.2
stack-data==0.6.3
sympy==1.12
tensorboard==2.17.0
tensorboard-data-server==0.7.2
tensorboard-logger==0.1.0
tensorboardX==2.6.2.2
tensorflow-probability==0.25.0
tensorstore==0.1.73
threadpoolctl==3.5.0
toolz==1.0.0
tornado==6.4.1
tqdm==4.66.4
traitlets==5.14.3
treescope==0.1.9
trimesh==4.6.6
typing-inspection==0.4.0
typing_extensions==4.13.1
tzdata==2024.1
urllib3==2.3.0
wandb==0.25.1
wcwidth==0.2.13
Werkzeug==3.1.3
wrapt==1.17.2
wurlitzer==3.1.1
zipp==3.21.0
REQ

python -m pip install -r "${REQ_FILE}"
rm -f "${REQ_FILE}"

# JAXMARL is project code in this repo. Prefer local editable install if present;
# otherwise install the exact upstream commit from the exported environment.
if [ -d "JaxMARL" ]; then
  python -m pip install -e JaxMARL
else
  python -m pip install \
    "git+https://github.com/huiby23/JAXMARL.git@feebfc1ab57e74457a291678a54cf5d0faa51253#egg=jaxmarl"
fi

python - <<'PY'
import sys
import jax
import torch

print("python", sys.version.split()[0])
print("jax", jax.__version__)
print("jaxlib", jax.lib.__version__)
print("jax devices", jax.devices())
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY

echo
echo "Done. For ov2 runs, usually use:"
echo "  conda activate ${ENV_NAME}"
echo "  export PYTHONPATH=\$PWD/experiments:\$PWD/JaxMARL"
echo "  export XLA_PYTHON_CLIENT_PREALLOCATE=false"
