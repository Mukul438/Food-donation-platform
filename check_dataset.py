import os

def check_dataset(folder):
    print(f"Checking folder: {folder}")
    for category in os.listdir(folder):
        category_path = os.path.join(folder, category)
        if os.path.isdir(category_path):
            num_images = len([
                f for f in os.listdir(category_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            print(f"  📂 {category}: {num_images} images")

print("\n--- Checking Training Data ---")
check_dataset("dataset/train")

print("\n--- Checking Validation Data ---")
check_dataset("dataset/val")
