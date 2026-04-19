"""Quick debug script to check encodec.encode() return shape."""
from audiocraft.models import AudioGen
import torch

model = AudioGen.get_pretrained("facebook/audiogen-medium")
encodec = model.compression_model
dummy = torch.randn(1, 1, 64000).cuda()
with torch.no_grad():
    encoded = encodec.encode(dummy)

print("type:", type(encoded))
print("len:", len(encoded))
if hasattr(encoded, "shape"):
    print("shape:", encoded.shape)
else:
    for i, item in enumerate(encoded):
        print(f"  [{i}] type={type(item)}", end="")
        if isinstance(item, tuple):
            for j, sub in enumerate(item):
                if hasattr(sub, "shape"):
                    print(f"  sub[{j}] shape={sub.shape}", end="")
                else:
                    print(f"  sub[{j}]={sub}", end="")
        elif hasattr(item, "shape"):
            print(f"  shape={item.shape}", end="")
        print()
