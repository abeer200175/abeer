import streamlit as st 
from keras.models import load_model

st.set_page_config('deprecation.showfileUploadEncoding',False)
set.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('kidney.h5')
    return model 
model=load_model()
st.write('# Kidney Disease Classifier')

file=st.file_uploader("Please upload an image", type=["jpg"])
import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data,model):

 size=(64,64)
 image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
 img=np.asarray(image)
 img_reshape=img[np.newaxis,...]
 prediction=model.predict(img_reshape)

 return prediction
if file is None :
 st.text("Please Upload an image file")
else:
   image=Image.open(file)
   st.image(image,use_column_width=True)
   predictions=import_and_predict(image,model)
   class_names=['Kidney Cyst','Normal Kidney','Kidney Stone','Kidney Tumor']
   string="its a "+class_names[np.argmax(predictions)]
   st.success (string)
   