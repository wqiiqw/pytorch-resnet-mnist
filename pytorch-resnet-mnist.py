#!/usr/bin/env python
# coding: utf-8

# # ResNet for MNIST in PyTorch 1.7
# For details see
# 
# https://zablo.net/blog/post/pytorch-resnet-mnist-jupyter-notebook-2021/
# ---
# 
# ## speical running notes
# * verified on Win10 + Cuda 11.6 + pip

# In[ ]:


# get_ipython().system('')


# In[ ]:


# get_ipython().system('nvidia-smi')


# In[ ]:


# get_ipython().system('pip install pytorch_lightning -i https://pypi.tuna.tsinghua.edu.cn/simple')
# get_ipython().system('pip install sklearn -i https://pypi.tuna.tsinghua.edu.cn/simple')


# In[ ]:


import torch
torch.__version__


# In[ ]:


from torchvision.models import resnet18
from torch import nn
from torch.utils.data import DataLoader


# In[ ]:


model = resnet18(num_classes=10)


# In[ ]:


model


# In[ ]:


model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


# In[ ]:


from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


# In[ ]:


train_ds = MNIST("mnist", train=True, download=True, transform=ToTensor())
test_ds = MNIST("mnist", train=False, download=True, transform=ToTensor())


# In[ ]:


train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64)


# In[ ]:


import pytorch_lightning as pl
#from pytorch_lightning.core.decorators import auto_move_data


# In[ ]:


class ResNetMNIST(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = resnet18(num_classes=10)
    self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    self.loss = nn.CrossEntropyLoss()

  #@auto_move_data
  def forward(self, x):
    return self.model(x)
  
  def training_step(self, batch, batch_no):
    x, y = batch
    logits = self(x)
    loss = self.loss(logits, y)
    return loss
  
  def configure_optimizers(self):
    return torch.optim.RMSprop(self.parameters(), lr=0.005)


# In[ ]:


model = ResNetMNIST()


# In[ ]:


trainer = pl.Trainer(
    gpus=1,
    max_epochs=1,
    progress_bar_refresh_rate=20
)


# In[ ]:


trainer.fit(model, train_dl)


# In[ ]:


trainer.save_checkpoint("resnet18_mnist.pt")


# In[ ]:


def get_prediction(x, model: pl.LightningModule):
  model.freeze() # prepares model for predicting
  probabilities = torch.softmax(model(x), dim=1)
  predicted_class = torch.argmax(probabilities, dim=1)
  return predicted_class, probabilities


# In[ ]:


from tqdm.autonotebook import tqdm


# In[ ]:


inference_model = ResNetMNIST.load_from_checkpoint("resnet18_mnist.pt", map_location="cuda")


# In[ ]:


true_y, pred_y = [], []
for batch in tqdm(iter(test_dl), total=len(test_dl)):
  x, y = batch
  true_y.extend(y)
  preds, probs = get_prediction(x, inference_model)
  pred_y.extend(preds.cpu())


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(true_y, pred_y, digits=3))


# In[ ]:




