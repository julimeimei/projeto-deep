from PIL import Image
import os

# Caminhos de entrada e saída
input_dir = 'data/train_images/'
output_dir = 'data/preprocessed_images/'

# Garantir que a pasta de saída existe
os.makedirs(output_dir, exist_ok=True)

# Redimensionar imagens
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    img = Image.open(img_path).resize((224, 224))
    img.save(os.path.join(output_dir, img_name))
