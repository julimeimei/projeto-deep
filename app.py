import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np

# Função para Focal Loss
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return focal_loss_fixed

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

# Calcular pesos para lidar com classes desbalanceadas
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data['diagnosis']),
    y=train_data['diagnosis']
)
class_weights = dict(enumerate(class_weights))

# Carregar a ResNet50 pré-treinada
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Adicionar camadas personalizadas para classificação
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # Adicionada regularização L2
x = Dropout(0.5)(x)
predictions = Dense(5, activation='softmax')(x)

# Criar o modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Descongelar mais camadas da ResNet50 para ajuste fino
for layer in base_model.layers[:-20]:  # Descongelando apenas as últimas 20 camadas
    layer.trainable = False

# Compilar o modelo com AdamW e Focal Loss
model.compile(optimizer=AdamW(learning_rate=0.00005, weight_decay=0.0001),
              loss=focal_loss(gamma=2., alpha=0.25),
              metrics=['accuracy'])

# Gerar os dados de treino e validação com aumento de dados ajustado
train_datagen = ImageDataGenerator(
    rescale=1./127.5,  # Normalizar para [-1, 1]
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    brightness_range=[0.8, 1.2]
)
val_datagen = ImageDataGenerator(rescale=1./127.5)

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

# Adicionar callbacks para melhorar o treinamento
checkpoint = ModelCheckpoint(
    'best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.75, patience=5, verbose=1, min_lr=1e-6
)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
)

# Treinar o modelo
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=40,  # Ajustado para 40 épocas
    callbacks=[checkpoint, reduce_lr, early_stopping],
    class_weight=class_weights
)

# Avaliar no conjunto de teste
test_generator = val_datagen.flow_from_dataframe(
    dataframe=test_data,
    directory='data/preprocessed_images',
    x_col='id_code',
    y_col='diagnosis',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Após a avaliação do modelo:
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=-1)

print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))