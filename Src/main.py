import torch
from data_preprocessing import prepare_data, get_dataloaders
from model import ResNet, ResidualBlock
from train import train_model
from evaluate import test_model
from visualize import visualize_feature_maps, plot_heatmap_with_image
from utils import generate_image_for_class, deconvolution

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths and class names
data_root = '/kaggle/input/architectural-heritage-elements-image64-dataset'
train_dir = 'train'
test_dir = 'test'
val_dir = 'val'
class_names = ['altar', 'apse', 'bell_tower', 'column', 'dome(inner)', 'dome(outer)', 'flying_buttress', 'gargoyle', 'stained_glass', 'vault']

# Prepare data
prepare_data(data_root, train_dir, test_dir, val_dir, class_names)

# Get dataloaders
batch_size = 128
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_dir, val_dir, test_dir, batch_size)

# Initialize model, loss, optimizer, and scheduler
model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
train_model(model, loss, optimizer, scheduler, num_epochs=50, train_dataloader=train_dataloader, val_dataloader=val_dataloader, device=device)

# Test the model
test_model(model, test_dataloader, loss, device)

# Visualize feature maps
for images, _ in test_dataloader:
    image = images[1]
    break
image = image.unsqueeze(0).to(device)
visualized_feature_maps = visualize_feature_maps(model, image, layer_id=1, device=device)
plot_heatmap_with_image(visualized_feature_maps, image, cmap='hot', alpha=0.8, vmin=0.2, vmax=0.8)

# Generate and deconvolve image for a specific class
target_class_index = 5  # Change this to the desired class index
generated_image = generate_image_for_class(model, target_class_index, device=device)
deconvolved_image = deconvolution(model, generated_image, target_class_index, device=device)

# Visualize the generated and deconvolved images
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(generated_image)
plt.title('Generated Image for Class: ' + class_names[target_class_index])
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(deconvolved_image)
plt.title('Deconvolved Image')
plt.axis('off')

plt.show()
