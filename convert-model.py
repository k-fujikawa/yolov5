import torch
from models.experimental import attempt_load

if __name__ == "__main__":
    MODELDIR = "output/8001_yolov5_0411"
    model = attempt_load(f"{MODELDIR}/best.pt", map_location="cpu")
    torch.save(model.state_dict(), f"{MODELDIR}/state_dict.pt")
