import time
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from oneformer import (
    add_common_config,
    add_oneformer_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup config
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(
        "configs/mapillary_vistas/convnext/oneformer_convnext_large_bs16_300k.yaml"
    )
    # Skip loading weights
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.IS_TRAIN = False
    cfg.MODEL.IS_DEMO = True
    cfg.freeze()

    # Build model
    model = build_model(cfg)
    model.eval()
    model.to(device)

    # Prepare random input
    H, W = 256, 320
    img = torch.rand(3, H, W, device=device) * 255.0
    batched_inputs = [{
        "image": img,
        "height": H,
        "width": W,
        "task": "The task is semantic",
    }]

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            model(batched_inputs)
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in tqdm(range(100)):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            model(batched_inputs)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    print(f"Mean inference time over 100 runs: {np.mean(times) * 1000:.2f} ms")

if __name__ == "__main__":
    main()
