# sparse net
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the network architecture
class SparseNet(nn.Module):
    def __init__(self, sparsity_rate, mutation_rate = 0.5):
        super(SparseNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.sparsity_rate = sparsity_rate
        self.mutation_rate = mutation_rate
        self.initialize_masks() # <== 1.initialize a network with random mask
   
    def forward(self, x):
        x = x.view(-1, 784)
        x = x @ (self.fc1.weight * self.mask1.to(x.device)).T + self.fc1.bias
        x = torch.relu(x)
        x = x @ (self.fc2.weight * self.mask2.to(x.device)).T + self.fc2.bias
        return x

    def initialize_masks(self):
        self.mask1 = self.create_mask(self.fc1.weight, self.sparsity_rate)
        self.mask2 = self.create_mask(self.fc2.weight, self.sparsity_rate)

    def create_mask(self, weight, sparsity_rate):
        k = int(sparsity_rate * weight.numel())
        _, indices = torch.topk(weight.abs().view(-1), k, largest=False)
        mask = torch.ones_like(weight, dtype=bool)
        mask.view(-1)[indices] = False
        return mask  # <== 1.initialize a network with random mask

    def update_masks(self):
        self.mask1 = self.mutate_mask(self.fc1.weight, self.mask1, self.mutation_rate)
        self.mask2 = self.mutate_mask(self.fc2.weight, self.mask2, self.mutation_rate)
        
    def mutate_mask(self, weight, mask, mutation_rate=0.5): # weight and mask: 2d shape
        # Find the number of elements in the mask that are True
        num_true = torch.count_nonzero(mask)

        # Compute the number of elements to mutate
        mutate_num = int(mutation_rate * num_true)
        
        # 3) pruning a certain amount of weights of lower magnitude
        true_indices_2d = torch.where(mask == True) # index the 2d mask where is true
        true_element_1d_idx_prune = torch.topk(weight[true_indices_2d], mutate_num, largest=False)[1]
        
        for i in true_element_1d_idx_prune:
            mask[true_indices_2d[0][i], true_indices_2d[1][i]] = False
        
        # 4) regrowing the same amount of random weights.
        # Get the indices of the False elements in the mask
        false_indices = torch.nonzero(~mask)

        # Randomly select n indices from the false_indices tensor
        random_indices = torch.randperm(false_indices.shape[0])[:mutate_num]

        # the elemnt to be regrow
        regrow_indices = false_indices[random_indices]
        for regrow_idx in regrow_indices:
            mask[tuple(regrow_idx)] = True
        
        return mask


# Set the device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset and move to the device
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('C:/Users/cv_ya/Desktop/git/Pruning/code/prune/example/data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

sparsity_rate = 0.5
model = SparseNet(sparsity_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

n_epochs = 10
for epoch in range(n_epochs):
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move the data to the device
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # print(f"Loss: {running_loss / (batch_idx+1)}")

    # Update masks
    model.update_masks() # generate a new mask based on the updated weights

    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss / (batch_idx+1)}")