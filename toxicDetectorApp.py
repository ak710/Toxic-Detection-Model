import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

MAX_FEATURES = 200000 # number of words in the vocab
model = tf.keras.models.load_model('Toxic detector.h5')
df = pd.read_csv('train.csv')
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,output_sequence_length=1800,output_mode='int')
X = df['comment_text']
vectorizer.adapt(X.values)

def score_comment(comment):
  vectorized_comment = vectorizer([comment])
  results = model.predict(vectorized_comment)
  return results

st.title("Predict how abusive your message is!")
col1, col2 = st.columns([3,1]) 
with col1:
  input = st.text_input(label="Please input your message here",label_visibility='collapsed', placeholder="Please input your message here")
with col2:
  button = st.button("Click here to predict")
if button and input:
    output = score_comment(input)
    fig, ax = plt.subplots()
    categories = df.columns[2:]
    toxicityPred = output[0]
    bar_colors = ['tab:red', 'tab:blue', 'tab:pink', 'tab:orange','tab:green','tab:purple']
    ax.bar(categories, toxicityPred,color=bar_colors)
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    res = True in (ele > 0.4 for ele in output[0])
    col3, col4, col5, col6, col7, col8 = st.columns(6) 
    with col3:
      st.metric(label="Toxicity", value="%.2f" % output[0][0])
    with col4:
      st.metric(label="Severe Toxicity", value="%.2f" % output[0][1])
    with col5:
      st.metric(label="Obscenity", value="%.2f" % output[0][2])
    with col6:
      st.metric(label="Threat", value="%.2f" % output[0][3])
    with col7:
      st.metric(label="Insult", value="%.2f" % output[0][4])
    with col8:
      st.metric(label="Identity Hate", value="%.2f" % output[0][5])
    if res:
      st.markdown("<h2 style='text-align: center; color: white;'>This is a toxic comment!</h2>", unsafe_allow_html=True)
    else:
      st.markdown("<h2 style='text-align: center; color: white;'>This is a not toxic comment!</h2>", unsafe_allow_html=True)