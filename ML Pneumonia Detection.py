import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

#größe der 'Paketen' von Bildern, die dem Modell beim Training zeitgleich gezeigt werden
batch_size = 32

#train_dir = os.path.join('chest_xray', 'train')
#val_dir = os.path.join('chest_xray', 'val')
#test_dir = os.path.join('chest_xray', 'test')

#Daten werden aus dem Verzeichnis mit Standardeinstellungen zur Preprocessing importiert. Es entsteht ein Tupel der Form (Bild, Label), wobei Bilder numpy arrays mit dem Shape (batch_size, image_size[0], image_size[1], num_channels) sind. Default: (auto. resize: 256x256px, mode: rgb, batchsize of 32, shuffle=True).
#Training-Daten werden benutzt um ein prädefiniertes Modell zu trainieren. Üblich ist es mind. 70% der Daten für diesen Zweck einzusetzen.
train_ds = tf.keras.utils.image_dataset_from_directory(
    '/Users/richard/Documents/Studium/1. Semester/Programmierung KI/Projekt/chest_xray/train',
    batch_size=batch_size,
    seed=123)

#Validation-Daten überwachen das neuronale Netzwerk während des Trainings. Nach jedem Durchlauf des Training-Datensatzes (epoch) wird die Genauigkeit des Modells anhand der Validation_Daten evaluiert. Üblich ist es 20% der Daten hierfür einzusetzen.
val_ds = tf.keras.utils.image_dataset_from_directory(
    '/Users/richard/Documents/Studium/1. Semester/Programmierung KI/Projekt/chest_xray/val',
    batch_size=batch_size,
    seed=123)

#Test-Daten sind Bilder, die das Netzwerk noch nie gesehen hat. Sie werden dem Modell nach Beendigung des Trainings präsentiert und erlauben eine endgültige Bewertung der Genauigkeit des neuronale Netzwerks. Üblich ist es 10% der Daten hierfür einzusetzen.
test_ds = tf.keras.utils.image_dataset_from_directory(
    '/Users/richard/Documents/Studium/1. Semester/Programmierung KI/Projekt/chest_xray/test',
    batch_size=batch_size,
    seed=123)

#Die class_names werden automatisch beim import von den Ordnernamen der Subdirectories generiert.
class_names = train_ds.class_names
print(class_names)

#Daten etwas visualisieren
#Ein Figure mit 25 der Bilder aus train_ds und den jeweiligen Labels wird erzeugt
plt.figure(figsize=(35, 35))
for images, labels in train_ds.take(1):
  for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]], fontsize=50)
    plt.axis("off")
#Erzeugtes Figure anzeigen
plt.show()


#Eine Methode um die Werte der RGB Farbkanäle vom range (0,255) zu range (0,1) abzuändern wird unter normalization_layer für einen späteren Einsatz abgelegt.
normalization_layer = tf.keras.layers.Rescaling(1./255)

#Es wird ein der Bilder angezeigt und dann der normalization_layer unterzogen, um dann mit min und max zu testen ob es auch wirklich funktioniert.
for images, labels in train_ds.take(1):
    image = images[0].numpy().astype("uint8")
    label = labels[0].numpy().astype("uint8")
plt.imshow(image)
result = normalization_layer(image)
plt.imshow(result)
print("Min and max pixel values:", result.numpy().min(), result.numpy().max())

#Eine Methode um die Training-Daten mit plausiblen, leicht geänderten Kopien (hier Flip und Rotation) zu ergänzen wird unter data_augmentation für einen späteren Einsatz abgelegt.
data_augmentation_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2)])

#Die Struktur der CNN wird vorweg festgelegt.
# Ein sequentielles Modell wird gewählt und mit Ebenen befüllt.
model = tf.keras.models.Sequential([
#wir rufen als erstes die oben definierten normalization und data_augmentation Ebenen auf.
    normalization_layer,
    data_augmentation_layer,

#Erster Convolutional Block
#Convolutional Ebene mit 32 Filters mit einem Suchfenster von 3x3px und einen Stride von 1px. ReLu wird eingesetzt: alle negative Werte der gefilterten Bilder werden genullt.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256,256,3)),
#die gefilterte Bilder werden in der Auflösung reduziert.
    tf.keras.layers.MaxPooling2D(2, 2),

#Zweiter Convolutional Block
#Convolutional Ebene mit 64 Filters mit einem Suchfenster von 3x3px und einen Stride von 1px. ReLu wird eingesetzt: alle negative Werte der gefilterten Bilder werden genullt.
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#die gefilterte Bilder werden in der Auflösung reduziert.
    tf.keras.layers.MaxPooling2D(2, 2),

#Dritter Convolutional Block
#Convolutional Ebene mit 128 Filters mit einem Suchfenster von 3x3px und einen Stride von 1px. ReLu wird eingesetzt: alle negative Werte der gefilterten Bilder werden genullt.
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#die gefilterte Bilder werden in der Auflösung reduziert.
    tf.keras.layers.MaxPooling2D(2, 2),

#Vierter Convolutional Block
#Convolutional Ebene mit 128 Filters mit einem Suchfenster von 3x3px und einen Stride von 1px. ReLu wird eingesetzt: alle negative Werte der gefilterten Bilder werden genullt.
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#die gefilterte Bilder werden in der Auflösung reduziert.
    tf.keras.layers.MaxPooling2D(2, 2),

#Eine Dropout-Ebene (Aktivierung 50% der Neurone wird auf null gesetzt, Auswahl random) wird eingesetzt um Over-fitting zu reduzieren.
    tf.keras.layers.Dropout(0.5),
#Die Anzahl der Neurone wird durch Konvergenz reduziert.
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
#Das Resultat sind 2 Outputs, ein pro Class, die angeben wie gut das Bild in der Class passt.
    tf.keras.layers.Dense(2)
])

#Das beschriebene Modell wird kompiliert mit einer Optimierung namens Adam. Die Funktion für die Berechnung des Verlustes (Loss) wird festgelegt. Accuracy wird als Metric angegeben.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Um Trainingszeiten durch vorheriges laden der Daten ins CPU zu reduzieren wird AUTOTUNE angewendet.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


#class weights
#for normal
#weight_for_0 = (1 / COUNT_NORMAL)*(TRAIN_IMG_COUNT)/2.0
#for pneumonia
#weight_for_1 = (1 / COUNT_PNEUMONIA)*(TRAIN_IMG_COUNT)/2.0

#class_weight = {0: weight_for_0, 1: weight_for_1}



#Die Anzahl der Epochs wird fesgelegt. Ein sog. Epoch besteht aus einmal einem Forward- und einmal einem Backward-Pass der gesamten Training- und Validation-Daten. Dabei werden die Gewichtungen der jeweiligen neuronalen Verbindungen angepasst.
epochs = 100

#Callbacks erlauben eine Möglichkeit in den Trainingsprozess einzugreifen. Hier wird EarlyStopping (frühzeitiges Stoppen) eingesetzt, um das Modell anzuhalten und das Training frühzeitig zu beenden, wenn keine Besserung der Metrics über 16 Epochs erzielt wurde.
callbacks = tf.keras.callbacks.EarlyStopping(patience=16)
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    # class_weight=class_weight,
    callbacks=callbacks)


#Das trainierte Modell wird im Directory gespeichert.
model.save('models/main neu.h5')

#die Metrics Accuracy und Loss für train_ds gegen val_ds während des Trainings werden geplottet und neben einander dargestellt.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#das trainierte Modell bekommt nun die Test-Daten zu sehen. Output sind Loss und Accuracy.
print(model.evaluate(test_ds))
