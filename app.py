import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import os

# Carregar o CSV
labels = pd.read_csv('data/train.csv')
labels['id_code'] = labels['id_code'].astype(str) + '.png'
labels['diagnosis'] = labels['diagnosis'].astype(int)

# Separar os dados em treino, validação e teste
train_data, test_data = train_test_split(labels, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

train_data['diagnosis'] = train_data['diagnosis'].astype(str)
val_data['diagnosis'] = val_data['diagnosis'].astype(str)
test_data['diagnosis'] = test_data['diagnosis'].astype(str)


# Carregar a ResNet50 pré-treinada
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Adicionar camadas personalizadas para classificação
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

# Criar o modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar as camadas da ResNet50
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Gerar os dados de treino e validação
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory='data/preprocessed_images',
    x_col='id_code',
    y_col='diagnosis',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data,
    directory='data/preprocessed_images',
    x_col='id_code',
    y_col='diagnosis',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Salvar o melhor modelo durante o treinamento
checkpoint = ModelCheckpoint(
    'best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1
)

# Treinar o modelo
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[checkpoint]
)
