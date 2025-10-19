import csv
import os

def generate_annotations(image_dir, data_filepath):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]  # Image paths from folder

    with open(data_filepath, mode ='r') as file:
        csvFile = csv.reader(file)
        next(csvFile, None)
        data_from_csv = []
        sorted_data = []

        for lines in csvFile:
            filename = lines[1]
            death_occurred = True if lines[5] == "Dead" else False

            if lines[4] != "'--": 
                survival_time = int(lines[4])
            else:
                survival_time = int(lines[40])

            
            data_from_csv.append([filename ,survival_time, int(death_occurred)])

    for image_path in image_paths:
        for data in data_from_csv:
            if data[0] in image_path:
                sorted_data.append([image_path, data[1], data[2]])
                break

    with open('annotations.csv', mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sorted_data)
    print("Finished generating annotations")


image_dir = r"./images"
data_filepath = r'./clinical.csv'
generate_annotations(image_dir, data_filepath) # Get annotations if needed