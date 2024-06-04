import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
import random
import shutil
import re
import cv2
#from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from unet import UNet
from torch import optim
import copy
import torch.nn as nn

#fix random seeds for reproducibility
seed_value = 1
np.random.seed(seed_value) # cpu vars
torch.manual_seed(seed_value) # cpu  vars
random.seed(seed_value) # Python
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value) # gpu vars
torch.backends.cudnn.deterministic = True  #needed
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# randomly sample image to show to get an intuition of the dataset.
img = np.asarray(Image.open("MLtestdata/area_3018_2013-04-26.jpg"))
# get the roof and solar mask associated with that image
roofmask = np.load("MLtestdata/area-3018-date-2013-04-26-labels-all-instance-mask_dk_2.npz")['dkmask']
solarmask = np.load("MLtestdata/area-3018-date-2013-04-26-labels-all-instance-mask_dk_3.npz")['dkmask']
#check the max and min value of the mask
print('roof mask: max value {}, min value {}'.format(np.amax(roofmask), np.amin(roofmask)))
print('solar mask: max value {}, min value {}'.format(np.amax(solarmask), np.amin(solarmask)))
#check if no label positions of roofmask and solar mask is consistent
roof_nolabel = roofmask == -1
solar_nolabel = solarmask == -1
print(roofmask.shape[0]*roofmask.shape[1], roofmask.shape)
print(np.sum(roof_nolabel), np.sum(solar_nolabel))
print('no label positions consistent', np.array_equal(roof_nolabel, solar_nolabel))
# let's see how they all line up. Remembering that pixels for the labels in the npz files can be >= 1 (present),
#0 (not present) or (-1) not labelled
#f, ax = plt.subplots(nrows=3, ncols=1, figsize=(20,60))
#ax[0].imshow(img)
#ax[1].imshow(roofmask, vmin=-1.0,vmax=1.0)
#ax[2].imshow(solarmask, vmin=-1.0,vmax=1.0)
#plt.show()

#remove images with no label or all labels are -1.
DATASET_DIR = "MLtestdata"
image_files = []
roofmask_files = []
solarmask_files = []
#build a list of image and mask file names, and rescale all images and masks to the 896*896
for file in os.listdir(DATASET_DIR):
    if file.endswith(".jpg"):
    	image_files.append(file)
    	img = np.asarray(Image.open(os.path.join(DATASET_DIR, file)))
    	if img.shape[0] != 896 or img.shape[1] != 896:
    	    img = cv2.resize(img, dsize=(896, 896), interpolation=cv2.INTER_CUBIC)
    	    cv2.imwrite(os.path.join(DATASET_DIR, file), img)
    	    print('...img saved',file)
    	   	

for file in os.listdir(DATASET_DIR):
    if file.endswith("-labels-all-instance-mask_dk_2.npz"):
    	roofmask_files.append(file)
    	roofmask_data = np.load(os.path.join(DATASET_DIR, file))['dkmask']
    	if roofmask_data.shape[0] != 896 or roofmask_data.shape[1] != 896:
    	    roofmask_data = cv2.resize(roofmask_data , dsize=(896, 896), interpolation=cv2.INTER_BITS)
    	    np.savez(os.path.join(DATASET_DIR, file), dkmask = roofmask_data)
    	    print('.......saved roof', file)

    	
for file in os.listdir(DATASET_DIR):
    if file.endswith("-labels-all-instance-mask_dk_3.npz"):
    	solarmask_files.append(file)
    	solarmask_data = np.load(os.path.join(DATASET_DIR, file))['dkmask']
    	if solarmask_data.shape[0] != 896 or solarmask_data.shape[1] != 896:
    	    solarmask_data = cv2.resize(solarmask_data, dsize=(896, 896), interpolation=cv2.INTER_BITS)
    	    np.savez(os.path.join(DATASET_DIR, file), dkmask = solarmask_data)
    	    print('.......saved solar', file)

print('num of images', len(image_files))    	
print('num of roofmask',len(roofmask_files))
print('num of solarmask',len(solarmask_files))


#find images with no label and move it out. 
# find images with no forground points and move it out.
#build training target [roof, solor], roof and solar can be 0 or 1, if there is no masks annotated, filled it with -1

IMAGES_WITHOUT_FG_LABEL = "imageswithoutfglabel"
if not os.path.exists(IMAGES_WITHOUT_FG_LABEL):
    os.mkdir(IMAGES_WITHOUT_FG_LABEL)
for img_file in image_files:
    img_name = img_file[:-4]
    img_name = re.sub("_", "-", img_name, count=1)   
    img_name = re.sub("_", "-date-", img_name, count=1)
    roofmask_name = img_name + "-labels-all-instance-mask_dk_2.npz"
    solarmask_name = img_name + "-labels-all-instance-mask_dk_3.npz"
    #print(roofmask_name)	
    if (not roofmask_name in roofmask_files) and (not solarmask_name in solarmask_files):
    	print(img_name + '.jpg has no labels, removed')
    	shutil.move(os.path.join(DATASET_DIR,img_file), os.path.join(IMAGES_WITHOUT_FG_LABEL,img_file))
    # check if both roof and solar mask have only -1 label
    elif (roofmask_name in roofmask_files) and (solarmask_name in solarmask_files):
    	roofmask_data = np.load(os.path.join(DATASET_DIR, roofmask_name))['dkmask']
    	solarmask_data = np.load(os.path.join(DATASET_DIR, solarmask_name))['dkmask']
    	if np.amax(roofmask_data) <= 0  and np.amax(solarmask_data) <= 0:
    		print('image with no foreground label removed')
    		shutil.move(os.path.join(DATASET_DIR,img_file), os.path.join(IMAGES_WITHOUT_FG_LABEL, img_file))
    		shutil.move(os.path.join(DATASET_DIR,roofmask_name), os.path.join(IMAGES_WITHOUT_FG_LABEL,roofmask_name))
    		shutil.move(os.path.join(DATASET_DIR,solarmask_name), os.path.join(IMAGES_WITHOUT_FG_LABEL,solarmask_name))
    	else:
    		combined_mask = np.stack((roofmask_data, solarmask_data), axis=-1)
    		np.savez(os.path.join(DATASET_DIR, roofmask_name[:-5]+'23.npz'), dkmask = combined_mask)
    		#print(os.path.join(DATASET_DIR, roofmask_name[:-5]+'23.npz'))
    		#print('combined mask saved', combined_mask.shape)
    elif (roofmask_name in roofmask_files) and (not solarmask_name in solarmask_files):
    	roofmask_data = np.load(os.path.join(DATASET_DIR, roofmask_name))['dkmask']
    	if np.amax(roofmask_data) <= 0:
    		shutil.move(os.path.join(DATASET_DIR,roofmask_name), os.path.join(IMAGES_WITHOUT_FG_LABEL, roofmask_name))
    		shutil.move(os.path.join(DATASET_DIR,img_file), os.path.join(IMAGES_WITHOUT_FG_LABEL, img_file))
    	else:
    		
    	#print('...', roofmask_data.dtype)
    		solarmask_data = -1 * np.ones((896,896),dtype=np.int8)
    	#print('--------', solarmask_data.dtype)
    		combined_mask = np.stack((roofmask_data, solarmask_data), axis=-1)
    		np.savez(os.path.join(DATASET_DIR, roofmask_name[:-5]+'23.npz'), dkmask = combined_mask)
    	
    elif (not roofmask_name in roofmask_files) and (solarmask_name in solarmask_files):
    	solarmask_data = np.load(os.path.join(DATASET_DIR, solarmask_name))['dkmask']
    	if np.amax(solarmask_data) <= 0:
    		shutil.move(os.path.join(DATASET_DIR,solarmask_name), os.path.join(IMAGES_WITHOUT_FG_LABEL,solarmask_name))
    		shutil.move(os.path.join(DATASET_DIR,img_file), os.path.join(IMAGES_WITHOUT_FG_LABEL, img_file))
    	else:
    	#print('...', solarmask_data.dtype)
    		roofmask_data = -1 * np.ones((896,896),dtype=np.int8)
    	#print('--------', roofmask_data.dtype)
    		combined_mask = np.stack((roofmask_data, solarmask_data), axis=-1)
    		np.savez(os.path.join(DATASET_DIR, solarmask_name[:-5]+'23.npz'), dkmask = combined_mask)
    		
#explore num of samples for roof, mask, no,and -1
total_roof = 0
total_no_roof = 0
for roofmask_file in roofmask_files:
    if os.path.isfile(os.path.join(DATASET_DIR, roofmask_file)):    	
    	roofmask_data = np.load(os.path.join(DATASET_DIR, roofmask_file))['dkmask']
    #print('....dict', roofmask_data)
    #print('roof', roofmask_data.shape)
    	num_roof = np.sum(roofmask_data >= 1)
    	num_no_roof = np.sum(roofmask_data == 0)
    #num_unknown = np.sum(roofmask_data == -1)
    	total_roof = total_roof + num_roof
    	total_no_roof = total_no_roof + num_no_roof 

roof_pos_weight = total_no_roof / total_roof
print('number of roof {}, number of no roof {}, roof positive weight {}'.format(total_roof,total_no_roof, roof_pos_weight))

total_solar=0
total_no_solar=0
for solarmask_file in solarmask_files:
    if os.path.isfile(os.path.join(DATASET_DIR, solarmask_file)): 
    	solarmask_data = np.load(os.path.join(DATASET_DIR, solarmask_file))['dkmask']
    #print('solar', solarmask_data.shape)
    	num_solar = np.sum(solarmask_data >= 1)
    	num_no_solar = np.sum(solarmask_data == 0)
    #num_unknown = np.sum(solarmask_data == -1)
    	total_solar = total_solar + num_solar
    	total_no_solar = total_no_solar + num_no_solar
    	
solar_pos_weight = total_no_solar / total_solar
print('number of solar {}, number of no solar {}, solar positive weight {}'.format(total_solar,total_no_solar, solar_pos_weight))  

roof2solar_rate =  total_roof /  total_solar
roof_labels2solar_labels = (total_roof + total_no_roof) / (total_solar + total_no_solar)
print('roof to solar rate', roof2solar_rate)
print('roof label to solar label rate', roof_labels2solar_labels)	
    		
   		
#cleaned image files
cleaned_image_files = []
for file in os.listdir(DATASET_DIR):
    if file.endswith(".jpg"):
    	cleaned_image_files.append(file)

#randomly split dataset into 80% training and 20% validation
random.shuffle(cleaned_image_files)
train_list = cleaned_image_files[:int(0.8*len(cleaned_image_files))]
val_list = cleaned_image_files[int(0.8*len(cleaned_image_files)):]

#create dataset and dataloader for train and val
class NearMapDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_list, root_dir, augument=False):
        """
        Arguments:
            data_list (list of string): file name of each image.
            root_dir (string): Directory with all the images.
            augument (bool): whether perform random horizontal and vertical flip with 0.5 propability
        """
        self.data_list = data_list
        self.root_dir = root_dir
        self.augument = augument

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_list[idx]
        #read image
        image = np.asarray(Image.open(os.path.join(self.root_dir, img_name)))
        image= np.copy(image) #array is no writable, so copy it
        #read mask
        img_name = img_name[:-4] # obtain the string name without extension
        img_name = re.sub("_", "-", img_name, count=1)   
        img_name = re.sub("_", "-date-", img_name, count=1)
        mask_name = img_name + "-labels-all-instance-mask_dk_23.npz"
        mask_data = np.load(os.path.join(DATASET_DIR, mask_name))['dkmask']
	#convert all >=1 to ==1 for multi-label binary cross entropy loss
        positive_location = (mask_data >= 1) 
        mask_data = (~positive_location) * mask_data + positive_location
	#get locations of no label, will be used to mask the loss calculation on these locations
        nolabel_location = (mask_data==-1)
	#replace the -1 to 0.5 for compatible with BCE calculation
        mask_data = 0.5 * nolabel_location + (~nolabel_location)*mask_data
	# random horizontal and vertival flip
        if self.augument == True:
        	r_num1 = random.uniform(0, 1)
        	if r_num1 > 0.5:
        		image = np.flipud(image)
        		mask_data = np.flipud(mask_data)
        		nolabel_location = np.flipud(nolabel_location)
        	r_num2 = random.uniform(0, 1)
        	if r_num2 > 0.5:
        		image = np.fliplr(image)
        		mask_data = np.fliplr(mask_data)
        		nolabel_location = np.fliplr(nolabel_location)		
	#convert to tensor
        image = torch.from_numpy(image)
        image = torch.permute(image, (2,0,1))#change to channel first
        image = image.float()
        image = image / 255.0
        mask_data = torch.from_numpy(mask_data)
        mask_data = torch.permute(mask_data, (2,0,1))
        nolabel_location = torch.from_numpy(nolabel_location)	
        nolabel_location = torch.permute(nolabel_location, (2,0,1))

        return image, mask_data, nolabel_location

#some hyperparameters
batch_size = 2
epochs = 20
lr = 0.00001

train_set = NearMapDataset(train_list, DATASET_DIR)
val_set = NearMapDataset(val_list, DATASET_DIR)

print('train set images', len(train_set))
print('val set images', len(val_set))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

#create model, we have two classes	
model = UNet(n_channels=3, n_classes=2)    
model = model.to(device)
print(model)
	
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)


#we need to mask the no label location when calulating loss.
# since the number of roof is mucher more than solar, our
def point_seg_loss(output, mask_data, nolabel_location, roof_labels2solar_labels, roof_pos_weight, solar_pos_weight):
    #the output is (batch, 2, 896, 896), we calculate roof and solar seperately 
    #print('output dim', output.size())  
    roof_output = output[:,0,:,:]
    #print('roof output dm', roof_output.size())
    roof_mask_data = mask_data[:,0,:,:]
    #print('roof mask dim',roof_mask_data.size() )
    roof_nolabel_location = nolabel_location[:,0,:,:]
    roof_bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([roof_pos_weight]).to(device), reduction='none')#the loss is the same shape as the output
    #print('device', roof_output.get_device(), roof_mask_data.get_device())
    roof_bce_loss = roof_bce_criterion(roof_output, roof_mask_data)
    #print('roof bce loss', roof_bce_loss)
    #mask out the no label position, and calculate the loss only on label location.
    roof_final_loss = torch.sum(roof_bce_loss*(~roof_nolabel_location)) / (2*torch.sum(roof_mask_data==0) + 1) # sum, then divide by 2*(number of negative labels) as positive are weighted to match negative, add 1 to avoid divide by 0
    #print('roof_final_loss', roof_final_loss)
    
    #now loss for solar
    solar_output = output[:,1,:,:]
    solar_mask_data = mask_data[:,1,:,:]
    solar_nolabel_location = nolabel_location[:,1,:,:]
    solar_bce_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([solar_pos_weight]).to(device), reduction='none')#the loss is the same shape as the output
    solar_bce_loss = solar_bce_criterion(solar_output, solar_mask_data)
    #print('solar bce loss', solar_bce_loss)
    #mask out the no label position, and calculate the loss only on label position.
    solar_final_loss = torch.sum(solar_bce_loss*(~solar_nolabel_location)) /  (2*torch.sum(solar_mask_data) + 1)
    #print('solar_final_loss', solar_final_loss)
    
    total_loss = roof_final_loss + (torch.Tensor([roof_labels2solar_labels]).to(device)) * solar_final_loss
    #print('batch total loss', total_loss)

    return total_loss


best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict()) 
early_stopping_patience = 10 
no_improve_epochs = 0
train_loss = []
val_loss = []
#start training      
for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    batch_ind = 0
    for image, mask_data, nolabel_location in train_loader:
        optimizer.zero_grad() 
        image = image.to(device)
        mask_data = mask_data.to(device)
        nolabel_location = nolabel_location.to(device)
        output = model(image)
        loss = point_seg_loss(output, mask_data,nolabel_location, roof_labels2solar_labels, roof_pos_weight, solar_pos_weight )
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()        
        print(f'Epoch [{epoch+1}/{epochs}], batch {batch_ind}/{len(train_loader.dataset)//batch_size}, Loss: {loss.item():.4f}')
        batch_ind +=1

    epoch_train_loss = running_train_loss / (len(train_loader.dataset) / batch_size)
    train_loss.append(epoch_train_loss)

    model.eval()
    val_running_loss = 0.0
    print('validating.............................')
    with torch.no_grad():
        for image, mask_data, nolabel_location in val_loader:
            image = image.to(device)
            mask_data = mask_data.to(device)
            nolabel_location = nolabel_location.to(device)
            output = model(image)       
            loss = point_seg_loss(output, mask_data,nolabel_location, roof_labels2solar_labels, roof_pos_weight, solar_pos_weight )
            val_running_loss += loss.item()

        epoch_val_loss = val_running_loss / (len(val_loader.dataset) / batch_size)
        scheduler.step(epoch_val_loss)
    val_loss.append(epoch_val_loss)
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

    if epoch_val_loss < best_val_loss:
        print(f'Validation Loss Decreased({best_val_loss:.6f}--->{epoch_val_loss:.6f}) \t Saving The Model')
        best_val_loss = epoch_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        state = {
            'net': model
        }
        torch.save(state, 'best.pt')
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    if no_improve_epochs > early_stopping_patience:
        print('Early stopping!')
        model.load_state_dict(best_model_wts)
        break
x = np.arange(epochs) 
train_loss = np.asarray(train_loss)   
val_loss = np.asarray(val_loss)
np.savez('traincurve.npz', train_loss=train_loss, val_loss=val_loss)


##evaluation
model.eval()

with torch.no_grad():
    roof_confusion_matrix = torch.zeros(2, 2)
    solar_confusion_matrix = torch.zeros(2, 2)
    for image, mask_data, nolabel_location in val_loader:
        image = image.to(device)
        mask_data = mask_data.to(device)
        nolabel_location = nolabel_location.to(device)
        output = model(image)
        outpred = torch.nn.functional.sigmoid(output) 
        #calculate confusion matrix
        roof_outpred = outpred[:,0,:,:]
        roof_pred_mask = roof_outpred > 0.5
        roof_label_mask = mask_data[:,0,:,:] 
        roof_label_location = ~(nolabel_location[:,0,:,:])
        roof_label_mask_value = roof_label_mask[roof_label_location]
        roof_pred_mask_value = roof_pred_mask[roof_label_location]
        for t, p in zip(roof_label_mask_value.view(-1), roof_pred_mask_value.view(-1)):
        	roof_confusion_matrix[t.long(), p.long()] += 1
               
        solar_outpred = outpred[:,1,:,:]  
        solar_pred_mask = solar_outpred > 0.5
        solar_label_mask =  mask_data[:,1,:,:] 
        solar_label_location = ~(nolabel_location[:,1,:,:])
        solar_label_mask_value = solar_label_mask[solar_label_location]
        solar_pred_mask_value = solar_pred_mask[solar_label_location]
        for t, p in zip(solar_label_mask_value.view(-1), solar_pred_mask_value.view(-1)):
        	solar_confusion_matrix[t.long(), p.long()] += 1

roof_noroof_acc = (roof_confusion_matrix[0,0] + roof_confusion_matrix[1,1]) / torch.sum(roof_confusion_matrix)
solar_nosolar_acc = (solar_confusion_matrix[0,0] + solar_confusion_matrix[1,1]) / torch.sum(solar_confusion_matrix)
        	
print('roof confusion matrix', roof_confusion_matrix)
print('roof noroof acc',roof_noroof_acc)
print('solar confusion matrix', solar_confusion_matrix)
print('solar nosolar acc',solar_nosolar_acc)

##prediction
testimage_name = val_list[20]
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

cv2.imwrite('test.jpg', testimage)
cv2.imwrite('test_roof.jpg', roof_test)
cv2.imwrite('test_solar.jpg', solar_test)

    		
    
    
    

