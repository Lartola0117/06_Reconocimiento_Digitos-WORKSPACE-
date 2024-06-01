# carga y exploracion de dataset desde keras
from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# explorar los datos
x_train.shape
import matplotlib.pyplot as plt
%matpltlib inline

plt.matshow(x_train[9])

print (y_train.shape)
print (y_train[:10])

#preparando los datos para el entrenamiento

x_train_prep = x_train.reshape(60000,28*28)
x_train_prep = x_train_prep.astype("float32")/255
x_test_prep = x_test.reshape(10000,28*28)
x_test_prep = x_test_prep.astype("float32")/255

print (x_train_prep.shape)
print (x_test_prep.shape)

#Creando la arquitectura de la red
from tensorflow import keras
from tensorflow.keras import layers 

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(512,activation="relu"),
    layers.Dense(10,activation="softmax")
                         
])

# Salvando los pesos iniciales, para el entrenamiento
pesos_iniales=model.get_weights()

#Compilar la red

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Entrenar la red

import tensorflow as tf
tb_caliback=tf.keras.calibacks.TensorBoard(log_dir="../logs/rmsprop", histogram_freq=1)
model_history=model.fit(x_train_prep, y_train, epochs=5,batch_size=128,callbacks=[tb_caliback])


model.compile(optimizer="SGD",
              loss="sparse_categorial_crossentropy",
              metrics=["accuracy"])
tb_callback = tf.keras.callbacks.TensorBoard(log_dir="../logs/SGD",histogram_freq=1)
model.set_weights(pesos_iniales)
model_history=model.fit(x_train_prep, y_train, epochs=5, batch_size=128, callbacks=[tb_callback])


model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
tb_callback=tf.keras.callbacks.TensorBoard(log_dir="../logs/adam", histogram_freq=1)
model.set_weights(pesos_iniales)
model_history=model.fit(x_train_prep, y_train, epochs=5, batch_size=128, callbacks=[tb_callback])

# Evaluar el modelo en Test
loss_en_test, acc_en_test=model.evaluate(x_test_prep, y_test)

# Realizr predcciones
#1 para el primer dato en test
digitos_prueba=x_test_prep[:1]# para obtener solo el 0
predicciones=model.predict(digitos_prueba)

print(predicciones[0])
print(predicciones[0].argmax())
print(predicciones[0,predicciones[0].argmax()])
y_test[0]

#2 para todo el juego de datos en test

y_prima= model.predict(x_test_prep)

y_prima.shape
y_prima[0].shape
print(y_prima[0])

#3 obtener las etiquetas asociadas a las predicciones
import numpy as np
y_prima_etiquetas=[np.argmax(el_caso) for el_caso in y_prima]

#4 construir la matriz de predicciones

matriz_confusion=tf.math.confusion_matrix(labels=y_test, predictions= y_prima_etiquetas)
print(matriz_confusion)

# Guardar el modelo

from keras.models import load_model
model.save ('..//models/reconocimiento_digitos_dense.keras ')



