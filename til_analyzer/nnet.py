import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
import torch
import cv2

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

import nnet_survival_pytorch

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from PIL import Image
import pandas as pd
from model import DifferentResNet186
from CustomTransformations import ImageTransformations
from main_test import test_on_chip

import random
torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


# Dataset class
class SurvivalDataset(Dataset):
    def __init__(self, images, labels, survivals, censoreds, transform=transforms.Compose([transforms.ToTensor()])):
        self.images = images
        self.labels = labels
        self.survivals = survivals
        self.transform = transform #Transformations
        self.censoreds = censoreds

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image.convert("RGB"))
        
        survival_data = self.labels[idx]
        survival_data = torch.from_numpy(survival_data).float()
        
        return image, survival_data, self.survivals[idx], self.censoreds[idx]

def get_data(annotations_path):
    with open(annotations_path, mode ='r') as file:

        csvFile = csv.reader(file)
        next(csvFile, None)
        image_filenames = []
        survival_times = []
        death_occurreds = []

        for lines in csvFile:
            image_filenames.append(lines[0])
            survival_times.append(float(lines[1]))
            if int(lines[2]) == 0:
                death_occurreds.append(1)
            elif int(lines[2]) == 1:
                death_occurreds.append(0)
            else:
                death_occurreds.append(int(lines[2]))

    print("Loaded", len(image_filenames), "image files")
    print("Loaded", len(survival_times), "survival times")
    print("Loaded", len(death_occurreds), "events\n")
    return image_filenames, survival_times, death_occurreds

def get_weights_by_class(dataloader):
    weights_dict = {
        "alive_cens": 0,
        "dead" : 0
    }
    for _images, survial_array, _survivals, censoreds_batch in dataloader:
        for censored in censoreds_batch:
            if int(censored) == 1:
                weights_dict["dead"] += 1
            else:
                weights_dict["alive_cens"] +=1

    weights = []
    for class_type in weights_dict:
        weights.append(weights_dict[class_type] / len(dataloader.dataset))
    print("Weights: ")
    print(weights)
    print()
    return weights

def get_weights_by_image(image_tensor):
    num_nonblack_pixels_per_image = torch.sum(image_tensor != -1, dim=[1, 2, 3])
    weights = torch.zeros(image_tensor.size(dim = 0))

    image_size = image_tensor[0].size(dim = 1) * image_tensor[0].size(dim = 2)
    for index in range(len(image_tensor)):
        if float(num_nonblack_pixels_per_image[index] / image_size) < 0.2:
            weights[index] = 0.0
        else:
            weights[index] = num_nonblack_pixels_per_image[index] / image_size 
    return weights

def save_results(train_loss, val_loss, train_accuracy, vaild_accuracy, train_c_index, val_c_index, excel_filepath):
    data = {
        "Training Loss": train_loss,
        "Validation Loss": val_loss,
        "Training Accuracy": [train_accuracy],
        "Validation Accuracy": [vaild_accuracy],
        "Training C-index 3 years": train_c_index[0],
        "Validation C-index 3 years": val_c_index[0],
        "Training C-index 2 years": train_c_index[1],
        "Validation C-index 2 years": val_c_index[1],
        "Training C-index 1 years": train_c_index[2],
        "Validation C-index 1 years": val_c_index[2],
        
    }
    df_new = pd.DataFrame(data)
    
    
    if os.path.exists(excel_filepath):
        existing_df = pd.read_excel(excel_filepath)
        df_combined = pd.concat([existing_df, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_excel(excel_filepath, index=False)   

# Training loop
def train_model(model, train_loader, val_loader, 
                num_epochs, breaks, loss_fn, learning_rate, class_weights, save_dir, save_period = 50):
    
    n = 0
    excel_filepath = save_dir + r'\metrics'+str(n)+'.xlsx'
    while(os.path.exists(excel_filepath)):
        n += 1
        excel_filepath = save_dir + r'\metrics'+str(n)+'.xlsx'


    highest_train_accuracy = 0
    highest_val_accuracy = 0
    lowest_train_loss = 1000
    lowest_val_loss = 1000

    save_model_dir = save_dir + r'\epoch '    
    
    
    # --------- SHOW SAMPLE IMAGES -----------
    num_sample_images = 1
    dataiter = iter(train_loader)
    images, _labels, _survivals, _censored = next(dataiter)

    # num_nonblack_pixels_per_image = torch.sum(images != -1, dim=[1, 2, 3])
    # image_size = images[0].size(dim = 1) * images[0].size(dim = 2)
    # weights = num_nonblack_pixels_per_image / image_size / 0.9
    # print(weights)

    for i in range(num_sample_images):
        transform = transforms.ToPILImage()
        image = transform(images[i])
        image.show()
        print(images[i].shape)

    # --------- OPTIMIZER SETTINGS -----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.4,
        patience=15,
        threshold=2.0,
    )

    # --------- TRAINING LOOP --------------
    for epoch in range(1,num_epochs+1):
        print(f"Epoch {epoch}\n-------------------------------")
        model.train()
        for batch, (images, survival_array, survivals, censoreds) in enumerate(train_loader):
            images, survival_array = images.to(device), survival_array.to(device)
            
            # ------ Custom Weighting -------
            # Decrease the weight of an image if the image is mostly black
            weights = get_weights_by_image(images)

            # Two classes, alive and dead 
            # Alter each image weight by the class size if there are significantly difference bewteen number of alive and dead
            # censored is 0 for alive and 1 for dead 
            for index in range(len(censoreds)):
                weights[index] *= (class_weights[censoreds[index]]/0.5)


            pred = model(images)
            loss = loss_fn(pred, survival_array, weights)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss, current = loss.item(), (batch + 1) * len(images)
            
            print(f"Training loss: {loss:>7f}")
        lr_scheduler.step(loss)
        print(lr_scheduler.get_last_lr())


        if epoch % 10 == 0:
            print("Training Accuracy")
            train_accuracy, train_loss, train_c_index = calculate_c_index(train_loader, model, loss_fn, breaks)
            print("Validation Accuracy")
            vaild_accuracy, val_loss, val_c_index = calculate_c_index(val_loader, model, loss_fn, breaks)
        
            save_results(train_loss, val_loss, train_accuracy, vaild_accuracy, train_c_index, val_c_index, excel_filepath)
            generate_survival_graph_with_KMF(val_loader, model, epoch, "Val")
            generate_survival_graph_with_KMF(train_loader, model, epoch, "Train")
            test_on_chip(model, epoch)

            if save_period != 0 and epoch % save_period == 0:
                torch.save(model.state_dict(), save_model_dir + str(epoch) + ".pt")
            if train_accuracy > highest_train_accuracy:
                torch.save(model.state_dict(), save_model_dir + "best_train_acc.pt")
                highest_train_accuracy = train_accuracy
            if vaild_accuracy > highest_val_accuracy:
                torch.save(model.state_dict(), save_model_dir + "best_val_acc.pt")
                highest_val_accuracy = vaild_accuracy
            if train_loss < lowest_train_loss:
                torch.save(model.state_dict(), save_model_dir + "lowest_train_loss.pt")
                lowest_train_loss = train_loss
            if val_loss < lowest_val_loss:
                torch.save(model.state_dict(), save_model_dir + "lowest_val_loss.pt")
                lowest_val_loss = val_loss


def calculate_c_index(dataloader, model, loss_fn, breaks):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    test_loss=0

    # Calculate c_index
    failed_c_index = 0
    survivals_whole = []
    pred_whole3 = []
    pred_whole2_5 = []
    pred_whole2 = []
    pred_whole1_5 = []
    pred_whole1 = []
    cens_whole = []

    with torch.no_grad():
        for images, surv_array, survivals, censoreds in dataloader:
            images, surv_array = images.to(device), surv_array.to(device)
            
            pred = model(images)
            test_loss += loss_fn(pred, surv_array, get_weights_by_image(images)).item()

            correct += (pred.argmax(1) == surv_array.argmax(1)).type(torch.float).sum().item()

            pred_c_index = pred.float().to(device).squeeze().detach().cpu().numpy()

            pred_c_index3 = nnet_survival_pytorch.nnet_pred_surv(pred_c_index, breaks, 3*365).tolist()
            pred_c_index2_5 = nnet_survival_pytorch.nnet_pred_surv(pred_c_index, breaks, 2.5*365).tolist()
            pred_c_index2 = nnet_survival_pytorch.nnet_pred_surv(pred_c_index, breaks, 2*365).tolist()
            pred_c_index1_5 = nnet_survival_pytorch.nnet_pred_surv(pred_c_index, breaks, 2*365).tolist()
            pred_c_index1 = nnet_survival_pytorch.nnet_pred_surv(pred_c_index, breaks, 1*365).tolist()

            pred_whole3.extend(pred_c_index3)
            pred_whole2_5.extend(pred_c_index2_5)
            pred_whole2.extend(pred_c_index2)
            pred_whole1_5.extend(pred_c_index1_5)
            pred_whole1.extend(pred_c_index1)

            survivals_whole.extend(survivals)
            cens_whole.extend(censoreds)
    
    c_indexes_to_save = [0]*3

    # pred_whole contains all of the predictions as a list (this is done because 
    # dataloaders load images by batches but all of the predicitons must be in a single list to run concordance_index to compare with survivals)

    #

    try:
        c_index3 = concordance_index(survivals_whole, pred_whole3, event_observed=cens_whole)
        print(f"Average c-index at 3 years: {c_index3:>8f},")

        c_index2_5 = concordance_index(survivals_whole, pred_whole2_5, event_observed=cens_whole)
        print(f"Average c-index at 2.5 years: {c_index2_5:>8f}")

        c_index2= concordance_index(survivals_whole, pred_whole2, event_observed=cens_whole)
        print(f"Average c-index at 2 years: {c_index2:>8f},")

        c_index1_5= concordance_index(survivals_whole, pred_whole1_5, event_observed=cens_whole)
        print(f"Average c-index at 1.5 years: {c_index1_5:>8f},")

        c_index1= concordance_index(survivals_whole, pred_whole1, event_observed=cens_whole)
        print(f"Average c-index at 1 year: {c_index1:>8f},")

        c_indexes_to_save = [c_index3, c_index2, c_index1]
    except ZeroDivisionError:
        failed_c_index += 1

    test_loss /= size
    correct /= size
    
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, Failed c-index: {failed_c_index:>0.1f}")
    return correct, test_loss, c_indexes_to_save

def generate_survival_graph_with_KMF(dataloader, model, epoch, dataset_name, save_dir = "./results"):
    surv = np.array(data[1])
    cens = np.array(data[2])


    breaks=np.arange(0.,365.*3,365./7)
    for batch, (images, surv_array, _survivals, _censoreds) in enumerate(dataloader):
        images, surv_array = images.to(device), surv_array.to(device)
        y_pred = model(images).float().to(device).squeeze().detach().cpu().numpy()
        transform = transforms.ToPILImage()
        image = transform(images[0])
        #image.show()
        cv2.waitKey(0)

        pred_surv = nnet_survival_pytorch.nnet_pred_surv(y_pred, breaks, 3*365)

        for i in range(len(pred_surv)):
            if pred_surv[i] > 0.7:
                plt.plot(breaks, np.concatenate(([1],np.cumprod(y_pred[i]))),'go-', label="a", alpha=0.2)
            elif pred_surv[i] > 0.5:
                plt.plot(breaks, np.concatenate(([1],np.cumprod(y_pred[i]))),'yo-', label="b", alpha=0.2)
            elif pred_surv[i] > 0.3:
                plt.plot(breaks, np.concatenate(([1],np.cumprod(y_pred[i]))),'ro-', label="c", alpha=0.2)
            else:
                plt.plot(breaks, np.concatenate(([1],np.cumprod(y_pred[i]))),'ko-', label="d", alpha=0.05)

    kmf = KaplanMeierFitter()
    kmf.fit(surv, event_observed=cens)


    plt.plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate,color='k')
    plt.xticks(np.arange(0, 2000.0001, 200))
    plt.yticks(np.arange(0, 1.0001, 0.125))
    plt.xlim([0,1200])
    plt.ylim([0,1])
    plt.xlabel('Follow-up time (days)')
    plt.ylabel('Proportion surviving')
    plt.title('One covariate. Actual=black, predicted=blue/red.')

    os.makedirs(save_dir, exist_ok=True)
    save_filepath = save_dir + r"\_" + dataset_name + '_epoch_'+str(epoch)+'.png'

    plt.savefig(save_filepath)
    plt.cla()





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    save_model_dir = "./results/saved_models"
    os.makedirs(save_model_dir, exist_ok=True)

    breaks=np.arange(0.,365.*3,365./7)
    num_intervals=len(breaks)-1

    training_split = 175/216
    batch_size = 16
    epochs = 400
    
    learning_rate = 0.00025
    
    annotations_path = r'./annotations.csv'
    # dead = 1
    # censored = 0

    global data 
    data = get_data(annotations_path)
    training_size = int(len(data[0]) * training_split)
    
    
    train_images = data[0][:training_size]
    train_survival_times = np.array(data[1][:training_size])
    train_censored = np.array(data[2][:training_size])

    val_images = data[0][training_size:]
    val_survival_times = np.array(data[1][training_size:])
    val_censored = np.array(data[2][training_size:])

    # 0 = censored = alive
    # 1 = dead
    transformations = ImageTransformations().train_transformations
    surv_train=nnet_survival_pytorch.make_surv_array(train_survival_times, train_censored, breaks)
    train_dataset = SurvivalDataset(train_images, surv_train, train_survival_times, train_censored, transformations)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    surv_val=nnet_survival_pytorch.make_surv_array(val_survival_times, val_censored, breaks)
    val_dataset = SurvivalDataset(val_images, surv_val, val_survival_times, val_censored, transformations)
    validation_loader = DataLoader(val_dataset, batch_size, shuffle=True)

    print(f"Images in train dataset: {len(train_loader.dataset)}")
    print(f"Images in validation dataset: {len(validation_loader.dataset)}\n")

    class_weights = [0.5,0.5]
    #get_weights_by_class(train_loader)
    print("Weights: " + str(class_weights))


    model = DifferentResNet186(num_intervals, 1024, 1024)
    model = model.to(device)

    loss_fn = nnet_survival_pytorch.surv_likelihood(num_intervals)
    train_model(model, train_loader, validation_loader,
                epochs, breaks, loss_fn, learning_rate, class_weights, save_model_dir, 20)

if __name__ == "__main__":
    main()

