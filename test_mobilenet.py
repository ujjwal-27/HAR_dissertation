from src.models import build_mobilenet

model = build_mobilenet()

print(model)

trainable = sum(
    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
)

print(f"\nTrainable Parameters : {trainable:,}")
