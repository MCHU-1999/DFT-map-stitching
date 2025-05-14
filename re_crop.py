from PIL import Image
import os


input_folder = 'maps_cropped_100px'   
output_folder = 'maps_cropped_20px' 
orig_size = 100
new_size = 20

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Ensure image is 50x50
        if img.size != (orig_size, orig_size):
            print(f"Skipping {filename}: not 50x50")
            continue

        # Compute box for centered crop
        left = (orig_size - new_size) // 2
        top = (orig_size - new_size) // 2
        right = left + new_size
        bottom = top + new_size
        cropped_img = img.crop((left, top, right, bottom))

        # Save to output folder with same filename
        output_path = os.path.join(output_folder, filename)
        cropped_img.save(output_path)

        print(f"Cropped and saved {filename}")

print("Done!")
