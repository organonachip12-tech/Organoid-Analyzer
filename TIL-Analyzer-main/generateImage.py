from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from PIL import Image
import pandas as pd

def draw_circle(x, y, radius, color, img):
    """
    Draws a red circle on a white canvas.
    
    :param x: X-coordinate of the circle's center
    :param y: Y-coordinate of the circle's center
    :param radius: Radius of the circle
    :param canvas_size: Tuple (width, height) for the canvas size
    """

    draw = ImageDraw.Draw(img)
    
    # Define the bounding box for the circle
    left_up = (x - radius, y - radius)
    right_down = (x + radius, y + radius)
    
    # Draw the circle
    draw.ellipse([left_up, right_down], fill=color, outline=color)


def draw_img(x_arr, y_arr, center_x, center_y, filename, center_radius=5018/2):
    canvas_size = (5235,5235)
    img = Image.new("RGB", canvas_size, "black")

    draw_circle(center_x, center_y, center_radius, "blue", img)

    for i in range(len(x_arr)):
        draw_circle(x_arr[i], y_arr[i], 20, "red", img)
    img.save(filename, "PNG")

    

def get_data(annotations_path, output_dir, caf_dosage, first_CART_concentration):

    with open(annotations_path, mode ='r') as file:
        csvFile = csv.reader(file)
        next(csvFile, None)[0]
        
        x = []
        y = []

        for lines in csvFile:
            if (lines[0] == "change"):
                first_CART_concentration = next(csvFile, None)[0] + ""
                continue
            elif (lines[1] == ""):
                filename = f"{output_dir}/{caf_dosage} {first_CART_concentration} {lines[0]}.png"
                print(filename)
                center_x = int(lines[3])
                center_y = int(lines[4])
                draw_img(x, y, center_x, center_y, filename)
                x = []
                y = []
                continue
            x.append(int(lines[1]))
            y.append(int(lines[2]))


def main():
    
    annotations_path = r".\chip annotations\image annotations\annotations3.csv"
    caf_concentration = "1.0 e6 CAF_AI"
    first_CART_concentration = "0.2 e7 CAR T" 

    chip_images_dir_path = "./Chip Images"
    os.makedirs(chip_images_dir_path, exist_ok=True)
    get_data(annotations_path, chip_images_dir_path, caf_concentration, first_CART_concentration)

if __name__ == "__main__":
    main()