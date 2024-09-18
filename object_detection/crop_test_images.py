import glob
import os
from tkinter import filedialog as fd
from PIL import Image

directory_path = fd.askdirectory()

Images = glob.glob(os.path.join(directory_path,"*.jpg"))

for image_file_and_path in Images:

    image_filename = os.path.basename(image_file_and_path)
    output_filename = os.path.join(r"datasets/SailbotVT-OG-Test-Cropped", image_filename)
    uncropped_image = Image.open(image_file_and_path)
    smaller_size = min(uncropped_image.size)
    larger_size = max(uncropped_image.size)

    # The argument to crop is a box : a 4-tuple defining the left, upper, right, and lower pixel positions.
    left = (larger_size-smaller_size)/2
    top = 0
    width = smaller_size
    height = smaller_size

    CroppedImg = uncropped_image.crop((left, top, width + left, height + top))

    CroppedImg.save(output_filename)
