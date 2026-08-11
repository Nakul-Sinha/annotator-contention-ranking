"""Run the contention-triage CV on a Modal A10G -- the same GPU the eval environment uses.

    modal run modal_app.py --tag g_r50 --backbone resnet50 --size 224 --epochs 30
"""
import os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = r"G:\ml\Latest_Chals\The Split Verdict Forecasting Asse\dataset"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy", "pandas", "scipy", "scikit-learn", "scikit-image",
                 "opencv-python-headless", "pillow", "timm")
    .add_local_dir(os.path.join(HERE, "code"), "/work/code")
    .add_local_dir(DATA, "/work/dataset")
)
app = modal.App("split-verdict")
vol = modal.Volume.from_name("sv-out", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/out": vol})
def run(args: list):
    import subprocess, sys
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
    os.makedirs("/out", exist_ok=True)
    cmd = [sys.executable, "/work/code/expcv.py", "--data", "/work/dataset", "--out", "/out"] + args
    print(" ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    vol.commit()
    return p.stdout[-14000:] + "\n=== STDERR ===\n" + p.stderr[-3000:]


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/out": vol})
def prep():
    """One-off: build the subject grouping + embeddings on the shared volume."""
    import subprocess, sys
    r = subprocess.run([sys.executable, "/work/code/make_groups.py", "/work/dataset", "/out", "141"],
                       capture_output=True, text=True)
    r2 = subprocess.run([sys.executable, "/work/code/handmodel.py", "--data", "/work/dataset",
                         "--out", "/out"], capture_output=True, text=True)
    vol.commit()
    return r.stdout[-3000:] + r.stderr[-2000:] + "\n=== hand ===\n" + r2.stdout[-4000:] + r2.stderr[-2000:]


@app.function(image=image, timeout=1800, volumes={"/out": vol})
def blend(extra: list):
    import subprocess, sys
    r = subprocess.run([sys.executable, "/work/code/blendall.py", "--out", "/out",
                        "--data", "/work/dataset"] + extra, capture_output=True, text=True)
    return r.stdout[-12000:] + r.stderr[-2000:]


@app.local_entrypoint()
def main(tag: str = "g1", backbone: str = "resnet50", size: int = 224, epochs: int = 30,
         extra: str = "", do_prep: bool = False):
    if do_prep:
        print(prep.remote())
        return
    args = ["--tag", tag, "--backbone", backbone, "--size", str(size), "--epochs", str(epochs)]
    args += [x for x in extra.split() if x]
    print(run.remote(args))
