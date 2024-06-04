import torch
import numpy as np
import cv2
from unet import UNet
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = "MLtestdata"
testimage_name = "area_6254_2017-08-28.jpg"
model = UNet(n_channels=3, n_classes=2)    
model = model.to(device)

ckpt = torch.load("best.pt")
ck = ckpt["net"]

model.load_state_dict(ck.state_dict())

##prediction
model.eval()
testimage = np.asarray(Image.open(os.path.join(DATASET_DIR, testimage_name)))
testimage = np.copy(testimage)
testimage = torch.from_numpy(testimage)
testimage = torch.permute(testimage, (2,0,1))
testimage = testimage.float()
testimage = testimage / 255.0
testimage = torch.unsqueeze(testimage, 0)
testimage = testimage.to(device)
test_out = model(testimage)
roof_test = test_out[0,0,:,:]
solar_test = test_out[0,1,:,:]

roof_test = torch.nn.functional.sigmoid(roof_test)
solar_test = torch.nn.functional.sigmoid(solar_test)

roof_test = roof_test > 0.5
solar_test = solar_test > 0.5
testimage = torch.squeeze(testimage)
testimage = torch.permute(testimage, (1,2,0))
#write predictions
testimage = testimage.detach().cpu().numpy()
roof_test = roof_test.detach().cpu().numpy()
solar_test = solar_test.detach().cpu().numpy()

print(testimage.shape)

roof_test = (roof_test*255.0).astype(np.uint8)
solar_test = (solar_test*255.0).astype(np.uint8)
testimage = (testimage*255.0).astype(np.uint8)

cv2.imwrite('test_sample.jpg', testimage)
cv2.imwrite('test_roof_sample.jpg', roof_test)
cv2.imwrite('test_solar_sample.jpg', solar_test)
