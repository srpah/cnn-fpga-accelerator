#Training LeNet-5 on MNIST

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)    # padding=0 (valid)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Data- ToTensor() scales pixels to [0,1] - value inside Q8.8 range
transform = transforms.ToTensor()
train_set = torchvision.datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_set  = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,   shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=1000, shuffle=False)


#Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = LeNet5().to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

def evaluate(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(test_set)

for epoch in range(5):
    net.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        F.cross_entropy(net(xb), yb).backward()
        opt.step()
    print(f"epoch {epoch}: float test acc = {evaluate(net):.4f}")

net.cpu().eval()
float_acc = evaluate(net.to('cpu')) if device == 'cpu' else None

# Save trained model
torch.save(net.state_dict(), "lenet.pth")
print("saved lenet.pth")

# Convert float values to Q8.8 fixed-point (scale by 256 and store as int16)
FRAC = 8
SCALE = 1 << FRAC                 # 256
QMIN, QMAX = -(1 << 15), (1 << 15) - 1

def to_fixed(arr):
    q = np.round(arr * SCALE).astype(np.int64)
    over = np.sum((q < QMIN) | (q > QMAX))
    if over:
        print(f"  WARNING: {over} values saturated in Q8.8 (consider Q9.7 or rescale)")
    return np.clip(q, QMIN, QMAX).astype(np.int16)

#testing quantization effect on accuracy of model
qnet = LeNet5()
qnet.load_state_dict(net.state_dict())
with torch.no_grad():
    for p in qnet.parameters():
        p.copy_(torch.from_numpy(to_fixed(p.numpy()).astype(np.float32) / SCALE))
qnet.eval()
correct = 0
with torch.no_grad():
    for xb, yb in test_loader:
        correct += (qnet(xb).argmax(1) == yb).sum().item()
print(f"quantized-weight test acc = {correct/len(test_set):.4f}")

#Exporting weights as C header
def emit_array(f, name, arr):
    flat = arr.flatten()
    f.write(f"const short {name}[{flat.size}] = {{\n")
    f.write(",".join(str(int(v)) for v in flat))
    f.write("\n};\n\n")

params = {
    "conv1_w": net.conv1.weight, "conv1_b": net.conv1.bias,
    "conv2_w": net.conv2.weight, "conv2_b": net.conv2.bias,
    "fc1_w":   net.fc1.weight,   "fc1_b":   net.fc1.bias,
    "fc2_w":   net.fc2.weight,   "fc2_b":   net.fc2.bias,
    "fc3_w":   net.fc3.weight,   "fc3_b":   net.fc3.bias,
}
with open("weights.h", "w") as f:
    f.write("// LeNet-5 weights, Q8.8 fixed point (value = raw / 256.0)\n\n")
    for name, p in params.items():
        emit_array(f, name, to_fixed(p.detach().numpy()))
print("wrote weights.h")

#Export one test image and label
img, label = test_set[0]
with open("test_image.h", "w") as f:
    f.write("// One MNIST test image, Q8.8 fixed point, shape 1x28x28\n\n")
    emit_array(f, "test_image", to_fixed(img.numpy()))
    f.write(f"const int test_label = {int(label)};\n")
print(f"wrote test_image.h (true label = {int(label)})")

# Generate reference outputs for each layer to validate FPGA inference
acts = {}
with torch.no_grad():
    x = img.unsqueeze(0)
    x = F.relu(net.conv1(x)); acts['after_conv1'] = x.numpy()
    x = F.max_pool2d(x, 2);   acts['after_pool1'] = x.numpy()
    x = F.relu(net.conv2(x)); acts['after_conv2'] = x.numpy()
    x = F.max_pool2d(x, 2);   acts['after_pool2'] = x.numpy()
    xf = x.view(1, -1)
    xf = F.relu(net.fc1(xf)); acts['after_fc1'] = xf.numpy()
    xf = F.relu(net.fc2(xf)); acts['after_fc2'] = xf.numpy()
    logits = net.fc3(xf);     acts['logits'] = logits.numpy()
np.savez("golden.npz", **acts)
print(f"wrote golden.npz  (predicted = {int(logits.argmax(1))}, true = {int(label)})")