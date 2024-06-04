import numpy as np
import matplotlib.pyplot as plt

train_history = np.load('traincurve.npz')['train_loss']
val_history = np.load('traincurve.npz')['val_loss']

x = np.arange(20)

plt.plot(x+1, train_history, 'r')
plt.plot(x+1, val_history, 'g')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "val"], loc="top right")
plt.show()
