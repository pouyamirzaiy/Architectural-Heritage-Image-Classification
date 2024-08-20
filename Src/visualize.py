            nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.deconv.to(device)

    def forward(self, x):
        return self.deconv(x)

def hook_fn(m, i, o):
    m.activations = o

def visualize_feature_maps(model, image, layer_id, device):
    model.layer1.register_forward_hook(hook_fn)
    model(image.to(device))
    activations = model.layer1.activations
    visualizer = Visualizer(activations.size(1), 3)
    activation = activations.unsqueeze(0) if activations.dim() == 3 else activations
    visualized_maps = visualizer(activation)
    return visualized_maps

def plot_heatmap_with_image(heatmap, image, cmap='viridis', alpha=0.6, vmin=None, vmax=None):
    heatmap_numpy = heatmap.cpu().detach().numpy()
    image_numpy = image.squeeze(0).cpu().detach().numpy()
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_numpy.transpose(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')
    num_channels = heatmap_numpy.shape[1]
    plt.subplot(1, 2, 2)
    for i in range(num_channels):
        heatmap_channel = heatmap_numpy[0, i, :, :]
        plt.imshow(heatmap_channel, cmap=cmap, interpolation='nearest', alpha=alpha, vmin=vmin, vmax=vmax)
        plt.title(f'Heatmap (Channel {i+1})')
        plt.axis('off')
    plt.show()
