

import ctypes
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudart64_110.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cublas64_11.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cublasLt64_11.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cufft64_10.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\curand64_10.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cusolver64_11.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cusparse64_11.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn64_8.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_adv_infer64_8.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_adv_train64_8.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_ops_infer64_8.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_cnn_train64_8.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_ops_infer64_8.dll')
ctypes.CDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_ops_train64_8.dll')
import numpy as np
seed = 2022
np.random.seed(seed)
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, log_loss,balanced_accuracy_score,recall_score,f1_score,accuracy_score
from keras import __version__ as keras_version
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
from keras_dataloader.datagenerator import DataGenerator

from keras import backend as K
from utils.utils import generate_data_csv,f1_m,precision_m, recall_m


print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
device_name = tf.test.gpu_device_name()
print(device_name)
random_state = 42
#columns=['ALB','BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
columns=['ALB','BET', 'DOL', 'LAG','MugilCephalus', 'OTHER','RhinobatosCemiculus','ScomberJaponicus','SHARK','TetrapturusBelone','Trout', 'YFT']
#columns = ['MugilCephalus','RhinobatosCemiculus','ScomberJaponicus','TetrapturusBelone','Trout']
IMG_COUNT = 224
IMG_SIZE = (IMG_COUNT, IMG_COUNT)
#print("device is: ",device)
#torch.cuda.set_per_process_memory_fraction(1.0, device=None) 

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
    


def create_model(): 
    #tf.keras.applications.efficientnet.EfficientNetB0(include_top=True).summary()
    
    #pretrained_model = tf.keras.applications.VGG19(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )
    #pretrained_model = tf.keras.applications.resnet.ResNet50(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )
    #pretrained_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=(299,299,3),weights='imagenet',include_top=False )
    #pretrained_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(244,244,3),weights='imagenet',include_top=False )
    #pretrained_model = tf.keras.applications.mobilenet.MobileNet(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )
    #pretrained_model = tf.keras.applications.xception.Xception(input_shape=(299,299,3),weights='imagenet',include_top=False )
    #for efficientnets pls check: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    pretrained_model = tf.keras.applications.efficientnet.EfficientNetB0(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )
    #pretrained_model = tf.keras.applications.efficientnet.EfficientNetB2(input_shape=(IMG_COUNT,IMG_COUNT,3),weights='imagenet',include_top=False )
    pretrained_model.summary()
    
    pretrained_model.trainable = False
    '''for i, layer in enumerate(pretrained_model.layers):
        if (layer.name.startswith('block7')) and not isinstance(layer, tf.keras.layers.BatchNormalization):
            print(i, layer.name)
            layer.trainable = True
        else:
            layer.trainable = False 
    '''

    print(pretrained_model.output)
    layers = [1024,256,64,len(columns)]
    sequential_layers = [pretrained_model,tf.keras.layers.GlobalAveragePooling2D(),tf.keras.layers.Flatten()]
    #sequential_layers = [pretrained_model,tf.keras.layers.Flatten()]
    top_dropout_rate = 0.5 
    for layer in layers:
        sequential_layers.append(tf.keras.layers.BatchNormalization())
        sequential_layers.append(tf.keras.layers.Dropout(top_dropout_rate))
        if layer != layers[-1]:
            sequential_layers.append(tf.keras.layers.Dense(layer, activation="relu"))    
        else:
            sequential_layers.append(tf.keras.layers.Dense(layer, activation="softmax", name="pred"))
        model = tf.keras.Sequential(sequential_layers)
    model.summary()
    
    optimizer = tf.keras.optimizers.RMSprop(momentum=0.0001)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics=['accuracy',f1_m,precision_m, recall_m],

    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

    return pretrained_model.name,model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv

def display_plot_for_history(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')


def run_cross_validation_create_models():
    
    # input image dimensions
    num_epoch = 100
    batch_size = 32

    #folder = 'NatureConservancyCropped'
    #folder = 'NatureConservancyCroppedAugmented'
    folder = 'NatureConservancyCroppedMerged'
    image_dir = '.'


    name,model = create_model()
    generate_data_csv(folder,columns, ['train','test'])
    #generate_data_csv(folder,['train'])

    train_data = pd.read_csv('data.csv')

    idg = tf.keras.preprocessing.image.ImageDataGenerator()
    train, test = train_test_split(train_data, test_size=0.10)
    train, val = train_test_split(train, test_size=2./9.)
    train_data_generator = idg.flow_from_dataframe(train, directory = image_dir,
                                x_col = "filename", y_col = "label",
                                class_mode = "categorical", shuffle = False,
                                target_size=(IMG_COUNT,IMG_COUNT),batch_size=batch_size)
            
    valid_data_generator  = idg.flow_from_dataframe(val, directory = image_dir,
                                x_col = "filename", y_col = "label",
                                class_mode = "categorical", shuffle = False,
                                target_size=(IMG_COUNT,IMG_COUNT),batch_size=batch_size)
    test_data_generator  = idg.flow_from_dataframe(test, directory = image_dir,
                                x_col = "filename", y_col = "label",
                                class_mode = "categorical", shuffle = False,
                                target_size=(IMG_COUNT,IMG_COUNT),batch_size=batch_size)
        
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(
        train_data_generator,
        epochs=num_epoch,
        validation_data=valid_data_generator,
        callbacks=[callback]
    )
    display_plot_for_history(history)
    #test data 

    predictions = model.predict(test_data_generator)
    #model.save('./models/model.h5')
    model.save('./models/modelX.h5')

    results = model.evaluate(test_data_generator)

    #extract labels
    Y_test = test_data_generator.classes
    
    predictions_labels = np.argmax(predictions,axis=1)
    conf_matrix = confusion_matrix(predictions_labels,Y_test)
    print("aciertos: ",np.trace(conf_matrix))
    print("total: ",np.sum(conf_matrix))
    print("test loss, test acc:", results)
    print("accuracy score is: ",accuracy_score(Y_test,predictions_labels  ))
    print("balanced accuracy score is: ",balanced_accuracy_score(Y_test,predictions_labels))
    print("f1-weighted score is: ",f1_score(Y_test,predictions_labels,average='weighted'))
    conf_matrix = conf_matrix *100/ conf_matrix.astype(float).sum(axis=0)
    conf_matrix = np.round(conf_matrix,decimals=3)
    print(conf_matrix)
    #np.savetxt("experimento3_conf_matrix.csv",conf_matrix,delimiter=",")
    #np.savetxt("experimento2_conf_matrix.csv",conf_matrix,delimiter=",")
    np.savetxt("experimentoX_conf_matrix.csv",conf_matrix,delimiter=",")
    

if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    run_cross_validation_create_models()
