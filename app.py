import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm

# Configurações
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
IMAGE_SIZE = 256

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        return
        
    # Crop baseado em threshold
    image_cropped = crop_image_from_gray(image)
    
    # Resize
    image_resized = cv2.resize(image_cropped, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Aplicar CLAHE em cada canal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b, g, r = cv2.split(image_resized)
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    
    # Merge canais processados
    processed_image = cv2.merge([b, g, r])
    
    cv2.imwrite(output_path, processed_image)

def process_dataset(input_dir, output_dir, df):
    os.makedirs(output_dir, exist_ok=True)
    
    def process_single_image(row):
        img_path = os.path.join(input_dir, row['id_code'])
        out_path = os.path.join(output_dir, row['id_code'])
        preprocess_image(img_path, out_path)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_single_image, 
                             [row for _, row in df.iterrows()]), 
                 total=len(df)))

def create_model():
    # Usar ResNetV2 como base
    base_model = ResNet50V2(weights='imagenet', 
                           include_top=False, 
                           input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    
    # Adicionar camadas customizadas
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.6)(x)  # Aumentar dropout
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.4)(x)  # Aumentar dropout
    predictions = Dense(5, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Congelar camadas iniciais
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    return model

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return focal_loss_fixed

# Carregar e preparar dados
labels = pd.read_csv('data/train.csv')

# Converter a coluna diagnosis para string ANTES do split
labels['diagnosis'] = labels['diagnosis'].astype(str)
labels['id_code'] = labels['id_code'].astype(str) + '.png'

# Processar imagens
process_dataset('data/train_images', 'data/preprocessed_images', labels)

# Split dados - agora com diagnosis já como string
train_data, test_data = train_test_split(labels, test_size=0.2, stratify=labels['diagnosis'])
train_data, val_data = train_test_split(train_data, test_size=0.1, stratify=train_data['diagnosis'])

# Calcular pesos das classes - usar os valores originais para o cálculo
original_diagnosis = labels['diagnosis'].astype(int)  # Converter temporariamente para int
class_weights = compute_class_weight('balanced',
                                   classes=np.unique(original_diagnosis),
                                   y=original_diagnosis)
class_weights = dict(enumerate(class_weights))

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./127.5,  # Normalizar para [-1, 1]
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./127.5)

# Preparar generators - agora com diagnosis como string
train_generator = train_datagen.flow_from_dataframe(
    train_data,
    directory='data/preprocessed_images',
    x_col='id_code',
    y_col='diagnosis',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    val_data,
    directory='data/preprocessed_images',
    x_col='id_code',
    y_col='diagnosis',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Criar e compilar modelo
model = create_model()
optimizer = AdamW(learning_rate=LEARNING_RATE, weight_decay=0.001)

# Habilitar mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

model.compile(
    optimizer=optimizer,
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    ModelCheckpoint('best_model.keras', 
                    monitor='val_accuracy',
                    save_best_only=True, verbose = 1),
    ReduceLROnPlateau(monitor='val_loss',
                      factor=0.75,
                      patience=3,
                      min_lr=1e-6),
    EarlyStopping(monitor='val_loss',
                  patience=5,
                  restore_best_weights=True, min_delta=0.001)
]

# Treinar modelo
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# Avaliar no conjunto de teste
test_generator = val_datagen.flow_from_dataframe(
    test_data,
    directory='data/preprocessed_images',
    x_col='id_code',
    y_col='diagnosis',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Avaliação final
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy:.4f}')