##Archivo de prueba para ver si se alimenta de la dataset
##Prueba de que sí funciona la dataset y el código
##mangos

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

# 1. Define la ruta base donde descomprimiste 'food-101, aquí están la carpeta meta e images'
DATA_ROOT = 'D:\\Datasets\\' 

# 2. Define las transformaciones (pre-procesamiento de imágenes)
transform = transforms.Compose([
    transforms.Resize(256),       # Cambia el tamaño para uniformidad
    transforms.CenterCrop(224),   # Recorta a 224x224, tamaño estándar para modelos pre-entrenados
    transforms.ToTensor(),        # Convierte la imagen a Tensor de PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Carga el conjunto de entrenamiento (PyTorch detectará las carpetas 'images' y 'meta')
try:
    train_dataset = datasets.Food101(
        root=DATA_ROOT,
        split='train',
        transform=transform,
        download=False  # ¡IMPORTANTE! Indica que ya tienes los archivos
    )

    test_dataset = datasets.Food101(
        root=DATA_ROOT,
        split='test',
        transform=transform,
        download=False
    )
    
    print(f"Dataset de entrenamiento cargado: {len(train_dataset)} muestras.")
    print(f"Dataset de prueba cargado: {len(test_dataset)} muestras.")

except RuntimeError as e:
    print(f"Error: No se pudo encontrar el dataset en la ubicación esperada.")
    print(f"Verifica que la carpeta 'food-101' en {DATA_ROOT} contenga 'images' y 'meta'.")
    print(e)