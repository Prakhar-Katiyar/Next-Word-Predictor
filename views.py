#!/usr/bin/env python
# coding: utf-8

# In[1]:


from django.shortcuts import render

# our home page view
def index(request):    
    return render(request, 'index.html')
from silence_tensorflow import silence_tensorflow

from tensorflow.keras.models import load_model

import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Load the model and tokenizer

model = load_model('network1.h5', compile = False)
tokenizer = pickle.load(open('tokenizer1.pkl', 'rb'))

def Predict_Next_Words(model, tokenizer, text):
    """
        In this function we are using the tokenizer and models trained
        and we are creating the sequence of the text entered and then
        using our model to predict and return the the predicted word.
    
    """
    for i in range(3):
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = np.array(sequence)
        
        preds = model.predict_classes(sequence)
#         print(preds)
        predicted_word = ""
        
        for key, value in tokenizer.word_index.items():
            if value == preds:
                predicted_word = key
                break
        
        return predicted_word
        

# our result page view
def expresion(request):


    text=request.POST['text1']
    text = text.split(" ")
    text = text[-1]

    text = ''.join(text)
    result=Predict_Next_Words(model, tokenizer, text)


    return render(request, 'index.html', {'result':result})

    


# In[ ]:




