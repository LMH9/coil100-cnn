import numpy as np
import pandas as pd
import seaborn as sns
import glob
import datetime
import cv2
import keras
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from keras.utils import img_to_array 
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


# Datenursprung definieren
path = 'input/coil-100/coil-100/*.png'
files = glob.glob(path)

# DataFrame konstruktieren
def constructDataframe(file_list):
    data = []
    for file in tqdm(file_list):
        data.append((file,file.split("/")[-1].split("__")[0]))
    return pd.DataFrame(data,columns=['path','label'])

print('DataFrame construction progress:')
df = constructDataframe(files)

# Daten sind gleichmäßig verteilt, s. Dokumentations aus dem Datensatz.
# Immer 72 Bilder pro Objekt --> daher keine Balancierung notwending

# Test/ Train Split --> 80:10:10 Split. Dazu splitten wir initial in 80:20 und danach den Test-Set abermals in 10:10
val_split = 0.4 
X_train, X_test, y_train, y_test = train_test_split(df.path, df.label, test_size=val_split,random_state=0,stratify= df.label)

print('\nX_train img to array progress:')
X_train = [img_to_array(cv2.imread(file).astype("float")/255.0) for file in tqdm(X_train.values)]
print('\nX_test img to array progress:')
X_test = [img_to_array(cv2.imread(file).astype("float")/255.0) for file in tqdm(X_test.values)]

# Umwandeln der String-Labels 'obj1' etc. in numerische Vektoren
encoder = LabelBinarizer()
y_train_categorical = encoder.fit_transform(y_train.values.reshape(-1,1))
y_test_categorical = encoder.transform(y_test.values.reshape(-1,1))

# Keras arbeitet mit NumPy Arrays
X_train=np.array(X_train)
X_test=np.array(X_test)

# Split Test Daten in Test und Validation Set
test_split = 0.5
X_test, X_validation, y_test_categorical, y_validation_categorical = train_test_split(X_test, y_test_categorical, test_size=test_split,random_state=0,stratify= y_test_categorical)

# CNN Model
def build(width, height, depth, classes):
    # Initializierung
    model = Sequential()
    # Erstes Conv Layer 30 Filter à 5x5 
    model.add(Conv2D(30, (5, 5), input_shape=(128, 128, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Zweites Conv Layer 15 Filter à 3x3.
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 20% der Neuronen des vorherigen Layers werden ausgelassen
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    # Output layer --> 100 Neuronen - 1 Neuron pro klassifizierbares Objekt
    model.add(Dense(classes, activation='softmax'))
    return model

EPOCHS = 25
INIT_LR = 0.001
BS = 32
model= build(128,128,3,encoder.classes_.__len__())
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

print('\nModel Schema:')
sequential_model_to_ascii_printout(model)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=4, verbose=1)
# Tensorboard
log_dir = "logs/coil/" + datetime.datetime.now().strftime("%Y%m%d") +f"TrainSize{str(1-val_split)}-ValSplit{str(test_split*val_split)}-TestSplit{str(val_split-(test_split*val_split))}"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# Callback zum Auswählen des besten Modells
mc = ModelCheckpoint(filepath = f"best_models/model-TrainSize{str(1-val_split)}-ValSplit{str(test_split*val_split)}-TestSplit{str(val_split-(test_split*val_split))}.h5", monitor = 'val_loss', mode = 'min', save_best_only = True, verbose = 1)

callbacks_list = [early_stopping, tensorboard_callback, mc]

# Data Augmentation - Daten werden leicht transfomiert. Dadurch werden neue Daten erzeugt. Hilft bei der Generalisierung
aug = ImageDataGenerator(rotation_range=30, 
                         width_shift_range=0.1, 
                         height_shift_range=0.1, 
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")

# Fit Model
model.fit(aug.flow(X_train, y_train_categorical, batch_size=BS), 
                           steps_per_epoch=len(X_train) // BS,
                           epochs=EPOCHS,
                           validation_data=(X_validation, y_validation_categorical),
                           verbose=1,
                           callbacks=callbacks_list)

# Lade bestes Modell
bestModel = load_model(f"best_models/model-TrainSize{str(1-val_split)}-ValSplit{str(test_split*val_split)}-TestSplit{str(val_split-(test_split*val_split))}.h5")
# Model Ausewertung
loss, accuracy = bestModel.evaluate(X_test,y_test_categorical, verbose=2)
print('Accuracy: %f' % (accuracy*100),'loss: %f' % (loss*100))

# Testen des Modells
print('\nStarting tests...')
print('Making X_test prediction')
prediction_test_c = bestModel.predict(X_test)
prediction_test = encoder.inverse_transform(prediction_test_c)

print('\nMaking X_train prediction')
prediction_train_c = bestModel.predict(X_train)
prediction_train = encoder.inverse_transform(prediction_train_c)

print('\nMaking X_validation prediction')
prediction_validation_c = bestModel.predict(X_validation)
prediction_validation = encoder.inverse_transform(prediction_validation_c)

# Heatmaps zur Evaluierung des Modells
def plot_cm(y,y_predict,classes,name):
    plt.figure(figsize = (100, 100))
    sns.heatmap(confusion_matrix(y,y_predict), 
            xticklabels = classes,
            yticklabels = classes)
    plt.title(name)
    plt.show()
    
plot_cm(prediction_test,encoder.inverse_transform(y_test_categorical),encoder.classes_,"Test accuracy")
plot_cm(prediction_validation,encoder.inverse_transform(y_validation_categorical),encoder.classes_,"Validation accuracy")
plot_cm(prediction_train,encoder.inverse_transform(y_train_categorical),encoder.classes_,"Train accuracy")