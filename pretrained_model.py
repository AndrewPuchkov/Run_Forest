import torch
import torchvision.transforms.v2 as T
import segmentation_models_pytorch as smp

checkpoint = "unet-resnet18.pt"

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=4,
    classes=2,
)

state_dict = torch.load(checkpoint, map_location="cpu")
model.load_state_dict(state_dict)
transforms = torch.nn.Sequential(T.Normalize(mean=[0.0], std=[255.0]))