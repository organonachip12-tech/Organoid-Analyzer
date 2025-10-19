import torch

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from PIL import Image
import pandas as pd
import umap

from lifelines.statistics import logrank_test

import nnet_survival_pytorch

from model import DifferentResNet186, Identity, GoodUmapandGraph
from CustomTransformations import ImageTransformations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset class
# SurvivalDataset outputs the images and the custom survival array (from nn_survival_pytorch.py) as a label for training the survival curve.
# RawSurvivalDataset outputs the images and the raw survival/censoring data.
class SurvivalDataset(Dataset):
    def __init__(self, data, labels, transform=transforms.Compose([transforms.ToTensor()])):
        self.data = data
        self.labels = labels
        self.transform = transform #Transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image.convert("RGB"))
        
        survival_data = self.labels[idx]
        survival_data = torch.from_numpy(survival_data).float()
        
        return image, survival_data
class RawSurvivalDataset(Dataset):
    def __init__(self, image_paths, surv_times, censored, transform=transforms.Compose([transforms.ToTensor()])):
        self.image_paths = image_paths
        self.surv_times = surv_times
        self.censored = censored
        self.transform = transform #Transformations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image.convert("RGB"))
        
        survival_data = self.surv_times[idx]

        censored_data = self.censored[idx]
        
        return image, survival_data, censored_data

def get_data(annotations_path):

    with open(annotations_path, mode ='r') as file:

        csvFile = csv.reader(file)
        image_filenames = []
        survival_times = []
        death_occurreds = []

        for lines in csvFile:
            image_filenames.append(lines[0])
            survival_times.append(float(lines[1]))
            death_occurreds.append(int(lines[2]))

    print("Loaded", len(image_filenames), "image files")
    print("Loaded", len(survival_times), "survival times")
    print("Loaded", len(death_occurreds), "events\n")
    return image_filenames, survival_times, death_occurreds

def save_results(survival_times, excel_filepath):
    df_new = pd.DataFrame(survival_times)
    
    
    if os.path.exists(excel_filepath):
        existing_df = pd.read_excel(excel_filepath)
        df_combined = pd.concat([existing_df, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_excel(excel_filepath, index=False)  


def generate_survival_graph_noKMF(low_dataloader, med_dataloader, high_dataloader, model, save_dir = './results/metrics'):
    os.makedirs(save_dir, exist_ok=True)

    n = 0
    excel_filepath = save_dir + r'\metrics'+str(n)+'.xlsx'
    while(os.path.exists(excel_filepath)):
        n += 1
        excel_filepath = save_dir + r'\metrics'+str(n)+'.xlsx'

    low_survivals_array = []
    med_survivals_array = []
    high_survivals_array = []

    surv_array = np.zeros((3, 15, 21))
    breaks=np.arange(0.,365.*3,365./7)


    is_first_entry = True
    for batch, (X, y) in enumerate(low_dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X).float().to(device).squeeze().detach().cpu().numpy()

        pred_surv = nnet_survival_pytorch.nnet_pred_surv(y_pred, breaks, 3*365)


        for i in range(len(pred_surv)):
            if is_first_entry:
                label = "(Low) 1.0e6 CAF"
                is_first_entry = False

            plt.plot(breaks, np.concatenate(([1],np.cumprod(y_pred[i]))),'ro-', label=label, alpha=0.15)
            low_survivals_array.append(np.concatenate(([1],np.cumprod(y_pred[i]))).tolist())
            label = ""

    surv_array[0] = low_survivals_array
    save_results(low_survivals_array, excel_filepath)

    is_first_entry = True
    for batch, (X, y) in enumerate(med_dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X).float().to(device).squeeze().detach().cpu().numpy()
        pred_surv = nnet_survival_pytorch.nnet_pred_surv(y_pred, breaks, 3*365)

        

        for i in range(len(pred_surv)):
            if is_first_entry:
                label = "(Med) 0.5e6 CAF"
                is_first_entry = False
            plt.plot(breaks, np.concatenate(([1],np.cumprod(y_pred[i]))),'bo-', label=label, alpha=0.15)
            med_survivals_array.append(np.concatenate(([1],np.cumprod(y_pred[i]))).tolist())
            label = ""

    surv_array[1] = med_survivals_array

    save_results(med_survivals_array, excel_filepath)


    is_first_entry = True
    for batch, (X, y) in enumerate(high_dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X).float().to(device).squeeze().detach().cpu().numpy()
        pred_surv = nnet_survival_pytorch.nnet_pred_surv(y_pred, breaks, 3*365)

        for i in range(len(pred_surv)):
            if is_first_entry:
                label = "(High) 0.0e6 CAF"
                is_first_entry = False
            plt.plot(breaks, np.concatenate(([1],np.cumprod(y_pred[i]))),'go-', label=label, alpha=0.15)
            high_survivals_array.append(np.concatenate(([1],np.cumprod(y_pred[i]))).tolist())
            label = ""

    surv_array[2] = high_survivals_array

    save_results(high_survivals_array, excel_filepath)

    median_list = [np.median(surv_array[i], axis=0) for i in range(3)]
    low_median = median_list[0]
    med_median = median_list[1]
    high_median = median_list[2]

    plt.plot(breaks, low_median,'ro-', label=label, alpha=1.0)
    plt.plot(breaks, med_median,'bo-', label=label, alpha=1.0)
    plt.plot(breaks, high_median,'go-', label=label, alpha=1.0)

    average_list = [np.average(surv_array[i], axis=0) for i in range(3)]
    low_average = average_list[0]
    med_average = average_list[1]
    high_average = average_list[2]

    plt.plot(breaks, low_average,'r-', label=label, alpha=1.0)
    plt.plot(breaks, med_average,'b-', label=label, alpha=1.0)
    plt.plot(breaks, high_average,'g-', label=label, alpha=1.0)
    
    print("Calculating Z-test of two proportions - P-values")
    p_value_categories = ["0.5_CAF-1.0_CAF_median", "0.0_CAF-0.5_CAF_median", "0.0_CAF-1.0_CAF_median",
                          "0.5_CAF-1.0_CAF_average", "0.0_CAF-0.5_CAF_average", "0.0_CAF-1.0_CAF_average"]

    p_values = np.ones((len(p_value_categories), len(median_list[0])))

    for interval in range(1,len(median_list[0])):
        low_median = median_list[0][0:interval]
        med_median = median_list[1][0:interval]
        high_median = median_list[2][0:interval]

        results = logrank_test(low_median,med_median)
        p_values[0][interval] = results.p_value

        results = logrank_test(med_median,high_median)
        p_values[1][interval] = results.p_value

        results = logrank_test(low_median,high_median)
        p_values[2][interval] = results.p_value


        low_average = average_list[0][0:interval]
        med_average = average_list[1][0:interval]
        high_average = average_list[2][0:interval]

        results = logrank_test(low_average,med_average)
        p_values[3][interval] = results.p_value
        
        results = logrank_test(med_average,high_average)
        p_values[4][interval] = results.p_value

        results = logrank_test(low_average,high_average)
        p_values[5][interval] = results.p_value
    
    save_results(["Medians"] ,excel_filepath)
    median_list = median_list[::-1]
    for i in range(len(median_list)):
        save_results(pd.DataFrame(np.append(median_list[i], str(i/2.0) + " e6 CAF")).transpose(), excel_filepath)
    
    save_results(["Averages"] ,excel_filepath)
    average_list = average_list[::-1]
    for j in range(len(average_list)):
        save_results(pd.DataFrame(np.append(average_list[j], str(j/2.0) + " e6 CAF")).transpose(), excel_filepath)

    save_results(["Z-test of two proportions p-values"] ,excel_filepath)

    save_results(["Medians"] ,excel_filepath)
    for k in range(3):
        save_results(pd.DataFrame(np.append(p_values[k], p_value_categories[k])).transpose(), excel_filepath)

    save_results(["Average"] ,excel_filepath)
    for l in range(3,6):
        save_results(pd.DataFrame(np.append(p_values[l], p_value_categories[l])).transpose(), excel_filepath)



    plt.xticks(np.arange(0, 2000.0001, 200))
    plt.yticks(np.arange(0, 1.0001, 0.125))
    plt.xlim([0,700])
    plt.ylim([0,1])
    plt.xlabel('Time (days)')
    plt.ylabel('Survival Probability')
    plt.title("All Survivals")
    plt.legend(loc = "lower right")
    save_dir = r"./validate_survivals/Umap"
    os.makedirs(save_dir, exist_ok=True)
    save_filepath = save_dir + r'\survival_curve.png'
    
    plt.savefig(save_filepath)
    plt.cla()


    df = pd.DataFrame(columns=['Name', 'Age', 'City'])


def getUMAP_all(model, high_surv_dataloader, med_surv_dataloader, low_surv_dataloader, validation_dataloader):
    all_appearence_vectors = []
    appearance_vectors = []

    for images, survivals, censoreds in high_surv_dataloader:
        
        images = images.to(device)
        high_surv_appearance_vec = model(images).float().to(device).detach().cpu().numpy()

        for i in range(len(high_surv_appearance_vec)):
            appearance_vectors.append(high_surv_appearance_vec[i])
    all_appearence_vectors.append(appearance_vectors)
    print(len(all_appearence_vectors[0]))

    appearance_vectors = []
    for images, survivals, censoreds in med_surv_dataloader:
        images = images.to(device)
        med_surv_appearance_vec = model(images).float().to(device).detach().cpu().numpy()

        for i in range(len(med_surv_appearance_vec)):
            appearance_vectors.append(med_surv_appearance_vec[i])
    all_appearence_vectors.append(appearance_vectors)        
    print(len(all_appearence_vectors[1]))

    appearance_vectors = []
    for images, survivals, censoreds in low_surv_dataloader:
        images = images.to(device)
        low_surv_appearance_vec = model(images).float().to(device).detach().cpu().numpy()

        for i in range(len(low_surv_appearance_vec)):
            appearance_vectors.append(low_surv_appearance_vec[i])
    all_appearence_vectors.append(appearance_vectors)
    print(len(all_appearence_vectors[2]))

    died_but_lived_more_than_2_years = []
    died_but_lived_less_than_2_years = []
    alive_more_than_2_years = []
    alive_less_than_2_years = []
    
    # --------- SHOW SAMPLE IMAGES -----------
    num_sample_images = 0
    dataiter = iter(validation_dataloader)
    images, labels, _cens = next(dataiter)
    #images, labels = images.to(device), labels.to(device)
    for i in range(num_sample_images):
        transform = transforms.ToPILImage()
        image = transform(images[i])
        image.show()
        print(images[i].shape)


    for images, survivals, censoreds in validation_dataloader:
        images = images.to(device)

        val_surv_appearance_vec = model(images).float().to(device).detach().cpu().numpy()

        for vec_index in range(len(val_surv_appearance_vec)):

            # Check which group to add the patient to:
            # If died (censored = 1)
            if int(censoreds[vec_index]) == 1:
                # If died and lived longer than 2 years
                if survivals[vec_index] >= 1*365:
                    died_but_lived_more_than_2_years.append(val_surv_appearance_vec[vec_index])
                else:
                    died_but_lived_less_than_2_years.append(val_surv_appearance_vec[vec_index])

            # If still alive: (censored = 0)
            else:
                # If alive and lived longer than 2 years
                if survivals[vec_index] >= 1*365:
                    alive_more_than_2_years.append(val_surv_appearance_vec[vec_index])
                else:
                    alive_less_than_2_years.append(val_surv_appearance_vec[vec_index])

    all_appearence_vectors.append(alive_more_than_2_years)
    all_appearence_vectors.append(died_but_lived_more_than_2_years)
    all_appearence_vectors.append(alive_less_than_2_years)
    all_appearence_vectors.append(died_but_lived_less_than_2_years)

    print(len(died_but_lived_more_than_2_years))
    print(len(died_but_lived_less_than_2_years))
    print(len(alive_more_than_2_years))
    print(len(alive_less_than_2_years))


    print("Preparing Umap")
    all_appearence_vectors_flat = [item for sublist in all_appearence_vectors for item in sublist]
    print(len(all_appearence_vectors_flat))

    # -------------- UMAP --------------
    for num_neighbors in range(2,50):
        umap_model = umap.UMAP(n_neighbors=num_neighbors, min_dist=0.5, n_components=2, random_state=42)
        embedding_2d = umap_model.fit_transform(all_appearence_vectors_flat)


        previous_interval = 0
        interval = len(all_appearence_vectors[0])

        colors = ["green", "blue", "red", "lime", "crimson", "darkgreen", "orange"]
        legend = ["Chip-0.0CAF", "Chip-0.5CAF", "Chip-1.0CAF", "Died-more2years", "Died-less2years", "alive-more2years", "alive-less2years"]

        for appear_index in range(0, len(all_appearence_vectors)):
            
            groupx = embedding_2d[:, 0][previous_interval:interval]
            groupy = embedding_2d[:, 1][previous_interval:interval]

            plt.scatter(groupx, groupy, c=colors[appear_index], label = legend[appear_index], alpha=1)
            
            previous_interval += len(groupx)
            
            if appear_index + 1 < len(all_appearence_vectors):
                interval += len(all_appearence_vectors[appear_index+1])
        

        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title(f'UMAP Projection of Extracted Features from All Images: neighbors_{num_neighbors}')
        plt.legend()

        
        save_dir = r"./validate_survivals/Umap/AllImages/UmapbyClass"
        os.makedirs(save_dir, exist_ok=True)
        save_filepath = save_dir + r'\all_images_neighbors_'+str(num_neighbors)+'.png'

        plt.savefig(save_filepath)
        plt.cla()

        data = list(zip(embedding_2d[:, 0], embedding_2d[:, 1]))
        inertias = []

        for num_cluster in range(1,11):
            kmeans = KMeans(n_clusters=num_cluster)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
            plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=kmeans.labels_)
            plt.title(f'UMAP colored with {num_neighbors} neighbors and {num_cluster} Kmeans clusters (All)')
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            save_dir = r"./validate_survivals/Umap/AllImages/UmapbyKmeans" + fr"/Neighbors_{num_neighbors}"
            os.makedirs(save_dir, exist_ok=True)
            save_filepath = save_dir + r'\kmeans_all_images_neighbors_'+str(num_neighbors)+'_clusters_'+str(num_cluster)+'.png'

            plt.savefig(save_filepath)
            plt.cla()

        plt.plot(range(1,11), inertias, marker='o')
        plt.title('Elbow method (All)')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        save_dir = r"./validate_survivals/Umap/AllImages/Kmeans"
        os.makedirs(save_dir, exist_ok=True)
        save_filepath = save_dir + r'\kmeans_all_images_neighbors_'+str(num_neighbors)+'.png'

        plt.savefig(save_filepath)
        plt.cla()

def getUMAP_only_chip(model, high_surv_dataloader, med_surv_dataloader, low_surv_dataloader):
    survivals_whole = []

    for images, survivals, censoreds in high_surv_dataloader:
        images = images.to(device)
        #predicted survival at the end of 3 years
        high_surv_appearance_vec = model(images).float().to(device).detach().cpu().numpy()

        for i in range(len(high_surv_appearance_vec)):
            survivals_whole.append(high_surv_appearance_vec[i])
    
    
    for images, survivals, censoreds in med_surv_dataloader:
        images = images.to(device)
        #predicted survival at the end of 3 years
        med_surv_appearance_vec = model(images).float().to(device).detach().cpu().numpy()

        for i in range(len(med_surv_appearance_vec)):
            survivals_whole.append(med_surv_appearance_vec[i])
    
    
    for images, survivals, censoreds in low_surv_dataloader:
        images = images.to(device)
        #predicted survival at the end of 3 years
        low_surv_appearance_vec = model(images).float().to(device).detach().cpu().numpy()

        for i in range(len(low_surv_appearance_vec)):
            survivals_whole.append(low_surv_appearance_vec[i])


    print(len(survivals_whole))
    print("Preparing Umap")
    
    for num_neighbors in range(2,15):
        print(f"Umap with {num_neighbors} neighbors")
        for min_distance_int in range(5, 75, 5):
            min_distance = min_distance_int/100.0

            umap_model = umap.UMAP(n_neighbors=num_neighbors, min_dist=min_distance, n_components=2, random_state=42)
            embedding_2d = umap_model.fit_transform(survivals_whole)

            #print(embedding_2d[:, 0])
            group1x = []
            group1y = []
            group2x = []
            group2y = []
            group3x = []
            group3y = []
            previous_interval = 0
            interval = len(high_surv_dataloader.dataset)

            for j in range(interval):
                group1x.append(embedding_2d[:, 0][j])
                group1y.append(embedding_2d[:, 1][j])
            
            previous_interval = interval
            interval += len(med_surv_dataloader.dataset)
            for k in range(previous_interval, interval):
                group2x.append(embedding_2d[:, 0][k])
                group2y.append(embedding_2d[:, 1][k])
            
            previous_interval = interval
            interval += len(low_surv_dataloader.dataset)
            for l in range(previous_interval, interval):
                group3x.append(embedding_2d[:, 0][l])
                group3y.append(embedding_2d[:, 1][l])


            # Plot the 2D UMAP projection
            plt.scatter(group1x, group1y, c='green', label="(High Response) 0.0 e6 CAF", alpha=1)
            plt.scatter(group2x, group2y, c='blue', label="(Med Response) 0.5 e6 CAF", alpha=0.5)
            plt.scatter(group3x, group3y, c='red', label="(Low Response) 1.0 e6 CAF",alpha=1)
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.title(f'UMAP Projection of Extracted Features from On-Chip Images: neighbors_{num_neighbors}')
            plt.legend()

            save_dir = r".\results\Umap\ChipImages\UmapbyClass"
            os.makedirs(save_dir, exist_ok=True)
            
            save_filepath = save_dir + r'\chip_neighbors_'+str(num_neighbors) + f"_{min_distance:.2f}" +'.png'

            plt.savefig(save_filepath)
            plt.cla()

            data = list(zip(embedding_2d[:, 0], embedding_2d[:, 1]))
            inertias = []

            for num_cluster in range(1,11):
                kmeans = KMeans(n_clusters=num_cluster)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
                plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=kmeans.labels_)
                plt.title(f'UMAP colored with {num_neighbors} neighbors and {num_cluster} Kmeans clusters')
                plt.xlabel('UMAP Dimension 1')
                plt.ylabel('UMAP Dimension 2')

                save_dir = r'.\results\Umap\ChipImages\UmapbyKmeans' + fr'\Neighbors_{num_neighbors}' 
                os.makedirs(save_dir, exist_ok=True)
                save_filepath = save_dir + r'\kmeans_chip_neighbors_'+str(num_neighbors) + f"_{min_distance:.2f}" + '_clusters_'+str(num_cluster)+'.png'

                plt.savefig(save_filepath)
                plt.cla()

            plt.plot(range(1,11), inertias, marker='o')
            plt.title('Elbow method')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            save_dir = r'.\results\Umap\ChipImages\Kmeans'
            os.makedirs(save_dir, exist_ok=True)
            save_filepath = save_dir + r'\kmeans_chip_neighbors_'+str(num_neighbors)+  f"_{min_distance:.2f}" +'.png'

            plt.savefig(save_filepath)
            plt.cla()



def main():
    breaks=np.arange(0.,365.*3,365./7)
    num_intervals=len(breaks)-1

    high_surv_annotations = r".\chip annotations\annotations_chip_high_surv.csv"
    med_surv_annotations = r".\chip annotations\annotations_chip_med_surv.csv"
    low_surv_annotations = r".\chip annotations\annotations_chip_low_surv.csv"
    train_annotations_path = r'.\annotations.csv'

    h_images, h_survival_times, h_censoreds = get_data(high_surv_annotations)
    m_images, m_survival_times, m_censoreds = get_data(med_surv_annotations)
    l_images, l_survival_times, l_censoreds = get_data(low_surv_annotations)

    transformations = ImageTransformations().validation_transformations
    
    h_surv_data=nnet_survival_pytorch.make_surv_array(np.array(h_survival_times), np.array(h_censoreds), breaks)
    h_data=SurvivalDataset(h_images, h_surv_data, transformations)
    h_data_loader = DataLoader(h_data, batch_size=16, shuffle=False)

    h_raw_dataset = RawSurvivalDataset(h_images, h_survival_times, h_censoreds, transformations)
    h_raw_data_loader = DataLoader(h_raw_dataset, batch_size=16, shuffle=False)


    m_surv_data=nnet_survival_pytorch.make_surv_array(np.array(m_survival_times), np.array(m_censoreds), breaks)
    m_data=SurvivalDataset(m_images, m_surv_data, transformations)
    m_data_loader = DataLoader(m_data, batch_size=16, shuffle=False)

    m_raw_dataset = RawSurvivalDataset(m_images, m_survival_times, m_censoreds, transformations)
    m_raw_data_loader = DataLoader(m_raw_dataset, batch_size=16, shuffle=False)

    l_surv_data=nnet_survival_pytorch.make_surv_array(np.array(l_survival_times), np.array(l_censoreds), breaks)
    l_data=SurvivalDataset(l_images, l_surv_data, transformations)
    l_data_loader = DataLoader(l_data, batch_size=16, shuffle=False)

    l_raw_dataset = RawSurvivalDataset(l_images, l_survival_times, l_censoreds, transformations)
    l_raw_data_loader = DataLoader(l_raw_dataset, batch_size=16, shuffle=False)

    batch_size = 8
    
    
    data = get_data(train_annotations_path)
    
    train_images = data[0]
    train_survival_times = np.array(data[1])
    train_censored = np.array(data[2])


    # Transformations
    transformations = ImageTransformations().umap_transformations

    raw_train_dataset = RawSurvivalDataset(train_images, train_survival_times, train_censored, transformations)
    raw_train_data_loader = DataLoader(raw_train_dataset, batch_size, shuffle=True)

    save_model_filepath = [
        ".\epoch 75.pt"
    ] 
    models = []

    for i in range(len(save_model_filepath)):
        models.append(GoodUmapandGraph(num_intervals))

        models[i].load_state_dict(torch.load(save_model_filepath[i], weights_only=True))
        models[i].to(device)
        models[i].eval()

    for model in models:
        generate_survival_graph_noKMF(l_data_loader, m_data_loader, h_data_loader, model)
        
        model.survival = Identity()
        model.sigmoid2 = Identity()

        getUMAP_only_chip(model,h_raw_data_loader, m_raw_data_loader, l_raw_data_loader)
        #getUMAP_all(model,h_test_loader, m_test_loader, l_test_loader, raw_train_data_loader)


    print("Done")

if __name__ == "__main__":
    main()

 


def validate_during_testing(low_dataloader, med_dataloader, high_dataloader, model, epoch):    
    low_survivals_array = []
    med_survivals_array = []
    high_survivals_array = []

    surv_array = np.zeros((3, 15, 21))
    breaks=np.arange(0.,365.*3,365./7)


    is_first_entry = True
    for batch, (X, y) in enumerate(low_dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X).float().to(device).squeeze().detach().cpu().numpy()

        pred_surv = nnet_survival_pytorch.nnet_pred_surv(y_pred, breaks, 3*365)


        for i in range(len(pred_surv)):
            if is_first_entry:
                label = "(Low) 1.0e6 CAF"
                is_first_entry = False

            plt.plot(breaks, np.concatenate(([1],np.cumprod(y_pred[i]))),'ro-', label=label, alpha=0.4)
            low_survivals_array.append(np.concatenate(([1],np.cumprod(y_pred[i]))).tolist())
            label = ""

    surv_array[0] = low_survivals_array

    is_first_entry = True
    for batch, (X, y) in enumerate(med_dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X).float().to(device).squeeze().detach().cpu().numpy()
        pred_surv = nnet_survival_pytorch.nnet_pred_surv(y_pred, breaks, 3*365)

        

        for i in range(len(pred_surv)):
            if is_first_entry:
                label = "(Med) 0.5e6 CAF"
                is_first_entry = False
            plt.plot(breaks, np.concatenate(([1],np.cumprod(y_pred[i]))),'bo-', label=label, alpha=0.4)
            med_survivals_array.append(np.concatenate(([1],np.cumprod(y_pred[i]))).tolist())
            label = ""

    surv_array[1] = med_survivals_array



    is_first_entry = True
    for batch, (X, y) in enumerate(high_dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X).float().to(device).squeeze().detach().cpu().numpy()
        pred_surv = nnet_survival_pytorch.nnet_pred_surv(y_pred, breaks, 3*365)



        for i in range(len(pred_surv)):
            if is_first_entry:
                label = "(High) 0.0e6 CAF"
                is_first_entry = False
            plt.plot(breaks, np.concatenate(([1],np.cumprod(y_pred[i]))),'go-', label=label, alpha=0.4)
            high_survivals_array.append(np.concatenate(([1],np.cumprod(y_pred[i]))).tolist())
            label = ""

    surv_array[2] = high_survivals_array
    plt.xticks(np.arange(0, 2000.0001, 200))
    plt.yticks(np.arange(0, 1.0001, 0.125))
    plt.xlim([0,1000])
    plt.ylim([0,1])
    plt.xlabel('Time (days)')
    plt.ylabel('Survival Probability')
    plt.title("All Survivals")
    plt.legend(loc = "lower right")

    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    save_filepath = save_dir + r'\chip_epoch_'+str(epoch)+'.png'

    plt.savefig(save_filepath)
    plt.cla()


    median_list = [np.median(surv_array[i], axis=0) for i in range(3)] 

    low_median = median_list[0]
    med_median = median_list[1]
    high_median = median_list[2]
    
    results = logrank_test(low_median,med_median)
    print("Low and Med LogRank p value: "+str(results.p_value))
    results = logrank_test(med_median,high_median)
    print("Med and High LogRank p value: "+str(results.p_value))
    results = logrank_test(low_median,high_median)
    print("Low and High LogRank p value: "+str(results.p_value))

def test_on_chip(test_model, epoch):
    breaks=np.arange(0.,365.*3,365./7)
    num_intervals=len(breaks)-1

    high_surv_annotations = r".\chip annotations\annotations_chip_high_surv.csv"
    med_surv_annotations = r".\chip annotations\annotations_chip_med_surv.csv"
    low_surv_annotations = r".\chip annotations\annotations_chip_low_surv.csv"

    h_images, h_survival_times, h_censoreds = get_data(high_surv_annotations)
    m_images, m_survival_times, m_censoreds = get_data(med_surv_annotations)
    l_images, l_survival_times, l_censoreds = get_data(low_surv_annotations)


    transformations = ImageTransformations().validation_transformations
    
    h_surv_data=nnet_survival_pytorch.make_surv_array(np.array(h_survival_times), np.array(h_censoreds), breaks)
    h_data=SurvivalDataset(h_images, h_surv_data, transformations)
    h_data_loader = DataLoader(h_data, batch_size=16, shuffle=False)
    h_raw_dataset = RawSurvivalDataset(h_images, h_survival_times, h_censoreds, transformations)


    m_surv_data=nnet_survival_pytorch.make_surv_array(np.array(m_survival_times), np.array(m_censoreds), breaks)
    m_data=SurvivalDataset(m_images, m_surv_data, transformations)
    m_data_loader = DataLoader(m_data, batch_size=16, shuffle=False)
    m_raw_dataset = RawSurvivalDataset(m_images, m_survival_times, m_censoreds, transformations)

    l_surv_data=nnet_survival_pytorch.make_surv_array(np.array(l_survival_times), np.array(l_censoreds), breaks)
    l_data=SurvivalDataset(l_images, l_surv_data, transformations)
    l_data_loader = DataLoader(l_data, batch_size=16, shuffle=False)
    l_raw_dataset = RawSurvivalDataset(l_images, l_survival_times, l_censoreds, transformations)

    validate_during_testing(l_data_loader, m_data_loader, h_data_loader, test_model, epoch)



