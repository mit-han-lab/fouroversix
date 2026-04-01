import subprocess

from ..resources import FOUROVERSIX_CACHE_PATH, app, cache_volume, get_image, hf_secret

img = get_image(dependencies=[], extra_pip_dependencies=["huggingface_hub[cli]"])


@app.function(
    image=img,
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH: cache_volume},
    secrets=[hf_secret],
)
def download_model(model_name: str) -> None:
    subprocess.run(["hf", "download", model_name], check=False)
