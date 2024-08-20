import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_kernels(tensor, num_cols=6):
    if not tensor.ndim == 4:
        raise Exception("Assumes a 4D tensor")
    if not tensor.shape[-1] == 3:
        raise Exception("Last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1 + num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def generate_image_for_class(model, target_class, num_steps=1000, lr=0.01, device='cuda'):
    input_image = torch.randn((1, 3, 224, 224), requires_grad=True, device=device)
    model.eval()
    model.to(device)
    optimizer = torch.optim.Adam([input_image], lr=lr)
    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(input_image)
        loss = -output[0, target_class]
        loss.backward()
        optimizer.step()
        input_image.data.clamp_(0, 1)
    generated_image = input_image.detach().cpu().squeeze().numpy().transpose(1, 2, 0)
    return generated_image

def deconvolution(model, generated_image, target_class_index, device='cuda'):
    model.eval()
    activation = None
    def hook(module, input, output):
        nonlocal activation
        activation = output
    target_layer = None
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            break
        target_layer = module
    target_layer.register_forward_hook(hook)
    input_tensor = torch.tensor(generated_image.transpose(2, 0, 1)).unsqueeze(0).to(device)
    output = model(input_tensor)
    if activation is None:
        raise ValueError("Activation is None. Ensure that the hook function is properly capturing activations.")
    decoder = nn.Sequential(
        nn.ConvTranspose2d(activation.size(1), 256, kernel_size=4, stride=2, padding=1).to(device),
        nn.ReLU(),
        nn.BatchNorm2d(256).to(device),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1).to(device),
        nn.ReLU(),
        nn.BatchNorm2d(128).to(device),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1).to(device),
        nn.ReLU(),
        nn.BatchNorm2d(64).to(device),
        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1).to(device),
        nn.ReLU()
    )
    normalized_activation = torch.nn.functional.normalize(activation)
    reconstructed_image = decoder(normalized_activation)
    return reconstructed_image.detach().cpu().squeeze().numpy().transpose(1, 2, 0)
