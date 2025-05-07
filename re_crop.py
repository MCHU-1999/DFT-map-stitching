from PIL import Image
import os

# Folder with original 50x50 images
input_folder = 'maps_cropped_50px'   
# Folder to save cropped 30x30 images
output_folder = 'maps_cropped_30px' 

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Ensure image is 50x50
        if img.size != (50, 50):
            print(f"Skipping {filename}: not 50x50")
            continue

        # Compute box for centered 30x30 crop
        left = (50 - 30) // 2
        top = (50 - 30) // 2
        right = left + 30
        bottom = top + 30
        cropped_img = img.crop((left, top, right, bottom))

        # Save to output folder with same filename
        output_path = os.path.join(output_folder, filename)
        cropped_img.save(output_path)

        print(f"Cropped and saved {filename}")

print("Done!")
