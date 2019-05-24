import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, InputLayer, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
%matplotlib inline




# epoch_num = 50 (다 50 찾아서 넣었음)
size = 224
class_weight = {0:1., 1:3.}
train_positive_data = "./data/train_positive"
train_negative_data = "./data/train_negative"
test_data = "./data/test"

tries = 1
list_layerNums = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
list_batch = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]

list_filters = [
                (64, 64, 96, 96, 128), 
                
                (32, 64, 96, 96, 128),
                (32, 64, 96, 96, 128),
                (32, 64, 96, 96, 128),
                (32, 64, 96, 96, 128),
                (64, 64, 128, 128, 128),
                (64, 64, 128, 128, 128),
                (64, 64, 128, 128, 128),
                (64, 64, 128, 128, 128),
                (64, 64, 128, 128, 128),
                (32, 64, 96, 96, 64),
                (32, 64, 96, 96, 64),
                (32, 64, 96, 96, 64),
                (32, 64, 96, 96, 64),
                (32, 64, 96, 96, 64),
                (64, 64, 96, 128, 128),
                (64, 64, 96, 128, 128),
                (64, 64, 96, 128, 128),
                (64, 64, 96, 128, 128),
                (64, 64, 96, 128, 128),
                ]
                
list_kernelsize = [
                    (3,2,2,2,2), 
                   
                   (3,3,2,2,2),
                   (3,3,2,1,1),
                   (3,3,3,3,3),
                   (3,2,2,2,1),

                   (3,3,2,2,1),
                   (3,3,2,2,2),
                   (3,3,2,1,1),
                   (3,3,3,3,3),
                   (3,2,2,2,1),
                   (3,3,2,2,1),
                   (3,3,2,2,2),
                   (3,3,2,1,1),
                   (3,3,3,3,3),
                   (3,2,2,2,1),
                   (3,3,2,2,1),
                   (3,3,2,2,2),
                   (3,3,2,1,1),
                   (3,3,3,3,3),
                   (3,2,2,2,1),
                   (3,3,2,2,1),
                   (3,3,2,2,2),
                   (3,3,2,1,1),
                   (3,3,3,3,3),
                   (3,2,2,2,1),
                   
                ]
list_records1 = []
list_records2 = []
list_records3 = []

list_1 = []
list_2 = []
list_3 = []
list_4 = []
list_5 = []

epoch_loss = []
epoch_val_loss = []
epoch_acc = []
epoch_val_acc = []

for e in range(0, 50):
    epoch_loss.append(0)
    epoch_val_loss.append(0)
    epoch_acc.append(0)
    epoch_val_acc.append(0)


def one_hot_label(img):
    label = img.split("_")[0]
    if label == "positive":
        ohl = np.array([1,0])
    elif label == "negative":
        ohl = np.array([0,1])
    return ohl

def get_train_data_with_label(train_data, size):
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size, size))
        train_images.append([np.array(img), one_hot_label(i)])
    return train_images

def get_test_data(test_data, size):
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size, size))
        test_images.append([np.array(img), i])
    return test_images

def split_train_images(negative_images, positive_images, fold):
    num_validation = 40
    validation_data = negative_images[num_validation*fold : num_validation*(fold+1)] + positive_images[num_validation*fold : num_validation*(fold+1)]
    train_data = negative_images[:num_validation*fold] + negative_images[num_validation*(fold+1):] + positive_images[:num_validation*fold] + positive_images[num_validation*(fold+1):]
    shuffle(train_data)
    shuffle(validation_data)
    return (train_data, validation_data)    


for t in range(0, tries):
    layerNum = list_layerNums[t]
    
    training_negative_images = get_train_data_with_label(train_negative_data, size)
    training_positive_images = get_train_data_with_label(train_positive_data, size)
    testing_images = get_test_data(test_data, size)

    train_loss, train_acc, val_loss, val_acc = (0, 0, 0, 0)


    list_last = []
    
    epoch_count = [0, 0, 0, 0, 0] #####초기화
    for e in range(0, 50):
        epoch_loss[e] = 0
        epoch_val_loss[e] = 0
        epoch_acc[e] = 0
        epoch_val_acc[e] = 0
    
    
    
    for fold in range(5):
        training_images, validation_images = split_train_images(training_negative_images, training_positive_images, fold)
        tr_img_data = np.array([i[0] for i in training_images])
        tr_lbl_data = np.array([i[1] for i in training_images])
        val_img_data = np.array([i[0] for i in validation_images])
        val_lbl_data = np.array([i[1] for i in validation_images])

        model = Sequential()
        model.add(InputLayer(input_shape = [size, size, 3]))
        
        
        
        for j in range(0, layerNum):
            model.add(Conv2D(filters=list_filters[t][j], kernel_size=list_kernelsize[t][j], strides=1, padding='same', activation='relu'))
            if (j != 4):
                model.add(MaxPool2D(pool_size=4, padding="same"))

        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        optimizer = Adam(lr=1e-3)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(patience=5) ############################################################ Patience
        checkpoint = ModelCheckpoint("./model", save_best_only=True)
        
        hist = model.fit (
            x = tr_img_data, 
            y = tr_lbl_data,
            epochs = 50, 
            batch_size = list_batch[t],
            validation_data = (val_img_data, val_lbl_data),
            callbacks = [early_stopping, checkpoint],
            class_weight = class_weight
        )
        model.summary()
        
        # history graph
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()
        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
        acc_ax.plot(hist.history['acc'], 'b', label='train acc')
        acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuracy')
        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')
        plt.show()
        
        fig = plt.figure(figsize=(14, 14))
        result = {
        "P-p" : 0,
        "P-n" : 0,
        "N-p" : 0,
        "N-n" : 0
         }
            
        for cnt, data in enumerate(testing_images):
            y = fig.add_subplot(40, 5, cnt+1)
            img = data[0]
            title = data[1]
            data = img.reshape(1, size, size, 3)
            model_out = model.predict([data])

            if np.argmax(model_out) == 1:
                str_label = "Negative"
            else:
                str_label = "Positive"

            if str_label == "Positive" and title.split("_")[0] == "positive":
                result["P-p"] += 1;
            elif str_label == "Positive" and title.split("_")[0] == "negative":
                result["P-n"] += 1;
            elif str_label == "Negative" and title.split("_")[0] == "positive":
                result["N-p"] += 1;
            elif str_label == "Negative" and title.split("_")[0] == "negative":
                result["N-n"] += 1;

            y.imshow(img)
            plt.title(str_label + ", " + title)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)
            
        print(result)
            
        accuracy = (result["P-p"]+result["N-n"])/(result["P-p"]+result["P-n"]+result["N-p"]+result["N-n"])
        # Accuracy = Pp+Np/Pp+Pn+Np+Nn 정확도(n을 N이라고, p를 P라고 예상한 총확률. 0.8 이상이 좋다.

        if(result["N-n"]+result["N-p"])==0:
            precision = 0 #0으로 나눌 경우 처리
        else:
            precision = result["N-n"]/(result["N-n"]+result["N-p"])
        # Precision = Nn/Nn+Pp Negative라고 예상한 데이터 중에서 실제로 Negative였던 게 어느 정도인지. 0.7 이상이 좋다.

        if(result["N-n"]+result["P-n"])==0:
            recall = 0 #0으로 나눌 경우 처리
        else:
            recall = result["N-n"]/(result["N-n"]+result["P-n"])
        # Recall(Sensitivity) = Nn/Nn+Pn 실제로 Negative인 데이터를 얼마나 찾아냈는지. 0.5 이상이 좋다.

        if(recall+precision)==0:
            f1 = 0
        else:
            f1 = 2*(recall*precision)/(recall+precision)
        
        print(" accuracy: " + str("%0.4f" % accuracy) + ", precision: " + str("%0.4f" % precision) + ", recall: " + str("%0.4f" % recall) + ", f1: " + str("%0.4f" % f1))



        for e in range(0, len(hist.history['loss'])): #각 epoch마다의 값을 저장
            epoch_count[fold] += 1 #이 fold에서 몇 epoch까지 갔는지 기록 (epoch 10에서 멈추면 10 기록)
            epoch_loss[e] += hist.history['loss'][e]
            epoch_val_loss[e] += hist.history['val_loss'][e]
            epoch_acc[e] += hist.history['acc'][e]
            epoch_val_acc[e] += hist.history['val_acc'][e]
        
        train_loss += hist.history['loss'][-1]
        val_loss += hist.history['val_loss'][-1]
        train_acc += hist.history['acc'][-1]
        val_acc += hist.history['val_acc'][-1]
        
        result_last = {
        "loss": "%0.4f" % hist.history['loss'][-1],
        "val_loss": "%0.4f" % hist.history['val_loss'][-1],
        "acc": "%0.4f" % hist.history['acc'][-1],
        "val_acc": "%0.4f" % hist.history['val_acc'][-1]
        }
        
        if(fold == 0):
            print(result_last)
            list_1.append(str(fold) + ". " + str(result_last)) # 결과 5개를 각각 list 1~5에 저장
        elif(fold == 1):
            print(result_last)
            list_2.append(str(fold) + ". " + str(result_last))
        elif(fold == 2):
            print(result_last)
            list_3.append(str(fold) + ". " + str(result_last))
        elif(fold == 3):
            print(result_last)
            list_4.append(str(fold) + ". " + str(result_last))
        elif(fold == 4):
            print(result_last)
            list_5.append(str(fold) + ". " + str(result_last))
    
        model.save("model_"+ str(t) + "_" + str(fold) + ".h5")

    result_average = {
        "avr_loss": "%0.4f" % (train_loss / 5),
        "avr_val_loss": "%0.4f" % (val_loss / 5),
        "avr_acc": "%0.4f" % (train_acc / 5),
        "avr_val_acc": "%0.4f" % (val_acc / 5)    
    }
    
    epoch_count_num = 0 #Epoch의 평균을 구하기 위한 코드 (10, 12, 12, 11, 10 Epoch까지 각각 갔으면 평균은 11)
    for n in range(5):
        epoch_count_num += epoch_count[n]
    avr_epoch_count = round(epoch_count_num/5)
    
    histo_loss = [] #epoch별 평균 값 저장을 위한 빈 리스트
    histo_val_loss = []
    histo_acc = []
    histo_val_acc = []
    
    for m in range(avr_epoch_count):
        to = 0
        for l in range(5): #                 epoch이 10, 12, 12, 11, 10까지 갔으면 10번째 epoch에서는 평균을 구할 때 값을 5로 나누고
            if(m < (epoch_count[l])): #    11번째 epoch에서는 값을 3로만 나눔, 이런 식, 근데 [l]-1인지 [l]인지 확실치 X 일단 [l]-1
#                 print(epoch_count[l]-1) #확인용
                to += 1
        if(to != 0): #평균 구하는 부분, to는 보통 5, 마지막쯤 가면 1~4의 숫자, 0으로는 나누지 않음
#             print(to) #몇으로 나눴는지 확인용
            histo_loss.append(epoch_loss[m]/to)
            histo_val_loss.append(epoch_val_loss[m]/to)
            histo_acc.append(epoch_acc[m]/to)
            histo_val_acc.append(epoch_val_acc[m]/to)
            
        
    fig, loss_ax = plt.subplots() #리스트로 평균 그래프 그려줌
    acc_ax = loss_ax.twinx()
    loss_ax.plot(histo_loss, 'y', label='avr train loss')
    loss_ax.plot(histo_val_loss, 'r', label='avr val loss')
    acc_ax.plot(histo_acc, 'b', label='avr train acc')
    acc_ax.plot(histo_val_acc, 'g', label='avr val acc')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    plt.show()
    
    
    
    
    
    list_records1.append("size: "+ str(size) + ", filter: " + str(list_filters[t]) + ", kernel size: " + str(list_kernelsize[t]))
    list_records2.append(str(result_average))
    list_records3.append(" accuracy: " + str("%0.4f" % accuracy) + ", precision: " + str("%0.4f" % precision) + ", recall: " + str("%0.4f" % recall) + ", f1: " + str("%0.4f" % f1))

#     list_records3.append(str(list_last))
    
    list_result_all = [] # 기록을 순서대로 출력 및 txt파일로 남김(덮어쓰기)
    for z in range(0, t+1):
        print(str(z+1))
        print(list_records1[z])
        list_result_all.append(str(list_records1[z]))
        print(list_records2[z])
        list_result_all.append(str(list_records2[z]))
        print(list_records3[z])
        
#         print(list_records3[z])
#         list_result_all.append(str(list_records3[z]))
        print(list_1[z])
        list_result_all.append(str(list_1[z])) #결과 5개 출력 및 리스트에 저장(txt용)
#         print(list_2[z])
#         list_result_all.append(str(list_2[z]))
#         print(list_3[z])
#         list_result_all.append(str(list_3[z]))
#         print(list_4[z])
#         list_result_all.append(str(list_4[z]))
#         print(list_5[z])
#         list_result_all.append(str(list_5[z]))

    result_txt = open("result.txt", "w")
    result_txt.write(str(list_result_all))
    result_txt.close()

