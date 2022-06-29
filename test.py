import torch
import torch.nn as nn

batch_size = 5
nb_classes = 2
in_features = 10

model = nn.Linear(in_features, nb_classes)
criterion = nn.CrossEntropyLoss()

x = torch.randn(batch_size, in_features)
target = torch.empty(batch_size, dtype=torch.long).random_(nb_classes)
print(target)

output = model(x)
print(output)
loss = criterion(output, target)
loss.backward()
