import torchHED
   
# process a single image file 
torchHED.process_file("/home/zcb/self_code_training/InTeX_self/2drefined_results/backpack/rendered_image.png", "/home/zcb/self_code_training/InTeX_self/2drefined_results/backpack/canny_image.png")

# process all images in a folder
# torchHED.process_folder("./input_folder", "./output_folder")

# process a PIL.Image loaded in memory and return a new PIL.Image
# img = PIL.Image.open("./images/sample.png")
# img_hed = torchHED.process_img(img)