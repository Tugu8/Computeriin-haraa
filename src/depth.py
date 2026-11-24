"""MiDaS загвар ачаалж, зурагнаас гүний зураг (depth map) гаргах туслах функцууд.

Хэрэглээ:
    from src.depth import load_model, predict_depth, read_image
    model, transform, device = load_model("DPT_Hybrid")
    img = read_image("data/images/room.jpg")
    depth = predict_depth(model, transform, img, device)

MiDaS-ийн гаргасан утга нь *relative inverse depth* — том утга = ойрхон,
жижиг утга = хол. Метрээр хэмжсэн бодит зай биш.
"""
import time

import cv2
import numpy as np
import torch

MODEL_TYPES = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]
HUB_REPO = "intel-isl/MiDaS"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_type="DPT_Hybrid", device=None):
    """torch.hub-аар MiDaS загвар болон түүнд тохирох transform-ыг ачаална."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type нь {MODEL_TYPES}-ийн аль нэг байх ёстой")
    device = device or get_device()

    if model_type == "MiDaS_small":
        # MiDaS_small дотроо EfficientNet-Lite3 backbone-ыг өөр hub репоноос татдаг.
        # Тэр репог урьдчилж "trusted" болгохгүй бол Colab/скрипт дээр input() асуугаад гацдаг.
        torch.hub.list("rwightman/gen-efficientnet-pytorch", trust_repo=True)

    model = torch.hub.load(HUB_REPO, model_type, trust_repo=True)
    model.to(device).eval()

    transforms = torch.hub.load(HUB_REPO, "transforms", trust_repo=True)
    # DPT загварууд 384x384, жижиг загвар 256x256 оролттой
    transform = transforms.small_transform if model_type == "MiDaS_small" else transforms.dpt_transform
    return model, transform, device


def read_image(path):
    """Зургийг RGB форматаар уншина (OpenCV анхдагчаар BGR уншдаг)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@torch.no_grad()
def predict_depth(model, transform, img_rgb, device=None):
    """Нэг RGB зурагнаас эх зурагтай ижил хэмжээтэй depth map (H x W, float32) буцаана."""
    device = device or next(model.parameters()).device
    batch = transform(img_rgb).to(device)

    prediction = model(batch)  # (1, h, w) — загварын оролтын хэмжээгээр
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    return prediction.cpu().numpy().astype(np.float32)


def normalize_depth(depth):
    """Depth map-ийг [0, 1] муж руу шилжүүлнэ (дүрслэхэд хэрэгтэй)."""
    d_min, d_max = float(depth.min()), float(depth.max())
    return (depth - d_min) / (d_max - d_min + 1e-8)


def depth_to_color(depth, cmap=cv2.COLORMAP_INFERNO):
    """Depth map-ийг өнгөт RGB зураг болгоно."""
    d8 = (normalize_depth(depth) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(d8, cmap), cv2.COLOR_BGR2RGB)


def benchmark(model, transform, img_rgb, device=None, runs=5, warmup=1):
    """Нэг зурагт зарцуулах дундаж хугацааг (секунд) хэмжинэ."""
    device = device or next(model.parameters()).device
    for _ in range(warmup):
        predict_depth(model, transform, img_rgb, device)
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        predict_depth(model, transform, img_rgb, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def count_params(model):
    return sum(p.numel() for p in model.parameters())
