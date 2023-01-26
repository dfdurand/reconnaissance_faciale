import numpy as np
import sklearn
import pickle
import cv2
import matplotlib.pyplot as plt



haar = cv2.CascadeClassifier('/home/durandroid/workspace/skull/Proj942/models/haarcascade_frontalface_default.xml')
model_svm = pickle.load(open("/home/durandroid/workspace/skull/Proj942/models/pca_svm2.pickle", mode = 'rb')) #machine learning model
pca_model = pickle.load(open('/home/durandroid/workspace/skull/Proj942/models/pca_dict2.pickle', mode='rb')) #pca dictionnary
model_pca = pca_model['pca'] #PCA model
mean_face = pca_model['mean-face'] #Mean Face


def faceRecognitionPipeline(filename):

    #img = cv2.imread('test/male_000281.jpg')
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.5, 3)
    predictions = []
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y), (x+w, y+h), (0,255,0), 2)
        roi = gray[y:y+h, x:x+w]
        #plt.imshow(roi, cmap='gray')
        #normalization
        roi = roi/255.0

        size = roi.shape[0]
        if size >= 100:
            roi_resize = cv2.resize(roi, (100,100), cv2.INTER_AREA) #retrecir
        else:
             roi_resize = cv2.resize(roi, (100,100), cv2.INTER_CUBIC) #agrandir

        roi_reshape = roi_resize.reshape(1, 10000)

        #image subtract

        roi_mean = roi_reshape - mean_face

        #08 eigen image (apply roi_mean)

        eigen_image = model_pca.transform(roi_mean)

        # 09 eigen image visualization
        eigen_img = model_pca.inverse_transform(eigen_image)

        #10 training

        results = model_svm.predict(eigen_image)
        score = model_svm.predict_proba(eigen_image)
        score_max = score.max()
        #print(results, score)

        #11 report
        text = "%s : %d"%(results[0], score_max*100) +"%"
        print(text)
        
        """
            ['Arthur' 'Arthur' 'Arthur' 'Arthur' 'Fabrice' 'Fabrice' 'Antonin'
 'Antonin' 'Arthur' 'Antonin' 'Antonin' 'Geoffrey' 'Antonin' 'Antonin'
 'Fabrice' 'Antonin' 'Arthur' 'Arthur' 'Antonin' 'Geoffrey' 'Arthur']
        """
        # if results[0] == "Arthur":
        #     color = (255,0,255)
        # elif results[0] == "Antonin":
        #     color = (0,255,255)
        # elif results[0] == "Geoffrey":
        #     color = (255,255,0)
        # elif results[0] == "Fabrice":
        #     color = (0,0,255)

        color = (255, 0, 255)

        cv2.rectangle(img,(x,y), (x+w, y+h), color, 2)
        cv2.rectangle(img,(x,y-60), (x+w, y), color, -1)
        cv2.putText(img,text, (x,y), cv2.FONT_HERSHEY_PLAIN,4, (255,255,255),3)
        output = {
            'roi': roi,
            'eigen': eigen_img,
            'prediction_name' : results[0],
            'score': score_max
        }

        predictions.append(output)

    return img, predictions
