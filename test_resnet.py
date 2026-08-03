from src.models.resnet import build_resnet

model = build_resnet()

print(model)

trainable = sum(
    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
)

print(f"\nTrainable Parameters : {trainable:,}")
