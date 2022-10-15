#|export
from fastai.vision.all import *
import gradio as gr
from fastbook import *

#|export
learn = load_learner('model.pkl')

#|export
categories = ('Rain','Rainbow')

def classify_image(im):
  pred, idx, probs = learn.predict(im)
  return dict(zip(categories, map(float, probs)))

#|export
image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['rain.jpg','rain2.jpg', 'rain3.jpg','rain4.jpg', 'rainbow.jpg','rainbow2.jpg','rainbow3.jpg']

intf = gr.Interface(fn=classify_image, inputs = image, outputs =label, examples = examples)
intf.launch(inline=False)