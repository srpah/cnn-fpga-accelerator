#converting golden.npz to a C header file
import numpy as np

g = np.load("golden.npz")

def emit(f, name, arr):
    flat = arr.flatten()
    f.write(f"const float {name}[{flat.size}] = {{\n")
    f.write(",".join(f"{v:.6f}" for v in flat))
    f.write("\n};\n\n")

with open("golden.h", "w") as f:
    f.write("// Golden FLOAT activations from PyTorch — reference for HLS C-sim.\n")
    f.write("// Compare with a tolerance (~0.02); fixed-point won't match bit-exact.\n\n")
    emit(f, "golden_conv1",  g["after_conv1"])  # shape 1x6x24x24 = 3456, NCHW flattened
    emit(f, "golden_logits", g["logits"])       # shape 1x10
print("wrote golden.h  (conv1 =", g["after_conv1"].size, "floats, logits = 10)")