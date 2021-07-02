########## imports ####################
from io import BytesIO
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions,
)

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import cv2
from PIL import Image
import io
from io import BytesIO
import base64
import numpy

#########################################
#architecture
#image
#top predicions
#heatmap select
#gradcam select

#implemented functions in app 
def get_predictions(img, model_name):
  """
  :param: Image, model 
  :return: DataFrame with top 10 predictions

  """
  #1 select model
  
  model_builder, img_size, preprocess_input, decode_predictions, last_conv_layer_name, target_size = architecture(model_name) #config['model_name']
  #2 load image
  #print(img, target_size)
  #img = load_image(img, target_size)
  #3 show image
  #test_show(img)
  #4 predict image
  df_prediction = predict_img(img, model_builder, preprocess_input, decode_predictions)
  return df_prediction

def get_gradcam_img(img, model_name, heatmap_style, gradcam_style, alpha):

  if gradcam_style == "grad_cam":
    
    #1 select model
    model_builder, img_size, preprocess_input, decode_predictions, last_conv_layer_name, target_size = architecture(model_name)
    #get image
    img_np = numpy.array(img)
    #create heatmap
    heatmap = make_gradcam_heatmap(img_np, model_builder, last_conv_layer_name, pred_index=None)
    #creat Gradcam (heatmap + img)
    gradcam_img = create_gradcam( img_np ,heatmap, alpha = alpha, style=heatmap_style)
    #test_show(gradcam_img)
    #plt.matshow(gradcam_img)

    #convert to base64
    #gradcam_export = image_to_base64(gradcam_img)

    return gradcam_img

  elif gradcam_style == "c_expl_heatmap":
    
    
    return counterfactual(img,heatmap_style, alpha, model_name)
  elif gradcam_style == "c_expl_blacked_out": #can be added
    blacked_counter = counterfactural_blacked_out(img, model_name, alpha)
    return blacked_counter
  else:
    return None

########### architecture selection ###### 
def architecture(architecture_name, **kwargs):
  """
  :param: wanted architecture
  :return: All important information about used architecture
  """

  global model_builder, img_size, preprocess_input, decode_predictions, last_conv_layer_name, target_size
  if architecture_name == "xception":
    model_builder = keras.applications.xception.Xception(**kwargs)
    img_size = (299, 299)
    preprocess_input = keras.applications.xception.preprocess_input
    decode_predictions = keras.applications.xception.decode_predictions

    last_conv_layer_name = "block14_sepconv2_act"  
    target_size = (299, 299, 3)

  elif architecture_name == "resnet50":
    model_builder = keras.applications.resnet50.ResNet50(**kwargs)
    img_size = (224, 224)
    preprocess_input = keras.applications.resnet50.preprocess_input
    decode_predictions = keras.applications.resnet50.decode_predictions

    last_conv_layer_name = "conv5_block3_out"
    target_size = (224, 224, 3)

  elif architecture_name == "mobilenetv2":
    model_builder = keras.applications.mobilenet_v2.MobileNetV2(**kwargs)
    img_size = (224, 224)
    preprocess_input = keras.applications.mobilenet_v2.preprocess_input
    decode_predictions = keras.applications.mobilenet_v2.decode_predictions

    last_conv_layer_name = "Conv_1"
    target_size = (224, 224, 3)

  elif architecture_name == "vgg16":
    model_builder = keras.applications.vgg16.VGG16(**kwargs)
    img_size = (224, 224)
    preprocess_input = keras.applications.vgg16.preprocess_input
    decode_predictions = keras.applications.vgg16.decode_predictions

    last_conv_layer_name = "block5_conv3"
    target_size = (224, 224, 3)

  return model_builder, img_size, preprocess_input, decode_predictions, last_conv_layer_name, target_size

########### methods - used in test ##############################

def load_image(filename, target_size):
  img = load_img(filename, target_size=target_size)
  return img

def test_show(img):
  #print(type(img))
  x = img_to_array(img)
  #print(type(x))
  #print(x.shape)

  plt.imshow(x/255.)
  plt.show()  

def get_img_array(img):
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def predict_img(img, model, preprocess_input, decode_predictions):
  img = get_img_array(img)
  img =  preprocess_input(img)
  # Print what the top predicted class is
  preds = model.predict(img)
  prediction_list = decode_predictions(preds, top=10)[0]
  df_prediction = pd.DataFrame(prediction_list, columns=['0', 'Predicted Object', 'Probability'])
  return df_prediction

def make_gradcam_heatmap(img, model, last_conv_layer_name, pred_index=None):
    # Remove last layer's softmax
    model.layers[-1].activation = None
    img_array = preprocess_input(get_img_array(img))
    
    
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy() 

def save_and_display_gradcam(img, heatmap, cam_path="/grad_cam_normal.jpg", alpha=0.9):
    # Load the original image
    img = img_to_array(img)
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    # cmaps['Miscellaneous'] = [
    #         'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
    #         'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
    #         'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
    #         'gist_ncar']
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    ## Save the superimposed image
    #superimposed_img.save(cam_path)
    #print("superimposed_img", superimposed_img)
    # return Grad CAM
    return superimposed_img

########################################

def counterfactual(img,heatmap_style, alpha, model_name):

  model_builder, img_size, preprocess_input, decode_predictions, last_conv_layer_name, target_size = architecture(model_name)

  #load image (np.array)
  img_np_ctfl = numpy.array(img)
  #ctcl img
  model = model_builder

  last_conv_layer = model.get_layer(last_conv_layer_name)
  last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

  classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
  x = classifier_input
  for layer_name in ["avg_pool", "predictions"]:
      x = model.get_layer(layer_name)(x)
  classifier_model = tf.keras.Model(classifier_input, x)

  with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(img_np_ctfl[np.newaxis, ...])
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]
  grads = tape.gradient(top_class_channel, last_conv_layer_output)
  pooled_grads = tf.reduce_mean(-1 * grads, axis=(0, 1, 2))
  last_conv_layer_output = last_conv_layer_output.numpy()[0]
  pooled_grads = pooled_grads.numpy()
  for i in range(pooled_grads.shape[-1]):
      last_conv_layer_output[:, :, i] *= pooled_grads[i]
  
  # Average over all the filters to get a single 2D array
  ctfcl_gradcam = np.mean(last_conv_layer_output, axis=-1)
  # Normalise the values
  ctfcl_gradcam = np.clip(ctfcl_gradcam, 0, np.max(ctfcl_gradcam)) / np.max(ctfcl_gradcam)
  ctfcl_gradcam = cv2.resize(ctfcl_gradcam, (224, 224))
  #heatmap from encountered
  multiobject_image = keras.preprocessing.image.img_to_array(img)
  heatmap = np.uint8(255 * ctfcl_gradcam)
  jet = cm.get_cmap(heatmap_style)
  jet_colors = jet(np.arange(256))[:, :3]
  jet_heatmap = jet_colors[heatmap]
  jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
  jet_heatmap = jet_heatmap.resize((multiobject_image.shape[1], multiobject_image.shape[0]))
  jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
  superimposed_img_ctfl = jet_heatmap * alpha + multiobject_image
  superimposed_img_ctfl = keras.preprocessing.image.array_to_img(superimposed_img_ctfl)
  #merge normal pic mit heatmap from encountered

  return superimposed_img_ctfl


def create_gradcam(img, heatmap, alpha, style):
  """
  style: string with heatmap style from dropdown
  """
  # Load the original image
  img = img_to_array(img)

  # Rescale heatmap to a range 0-255
  heatmap = np.uint8(255 * heatmap)

  # Use jet colormap to colorize heatmap
  jet = cm.get_cmap(style)
  # https://matplotlib.org/stable/tutorials/colors/colormaps.html
  # cmaps['Miscellaneous'] = [
  #         'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
  #         'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
  #         'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
  #         'gist_ncar']


  # Use RGB values of the colormap
  jet_colors = jet(np.arange(256))[:, :3]
  jet_heatmap = jet_colors[heatmap]

  # Create an image with RGB colorized heatmap
  jet_heatmap = array_to_img(jet_heatmap)
  jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
  jet_heatmap = img_to_array(jet_heatmap)

  # Superimpose the heatmap on original image
  superimposed_img = jet_heatmap * alpha + img
  superimposed_img = array_to_img(superimposed_img)

  ## Save the superimposed image
  #superimposed_img.save(cam_path)
  #print("superimposed_img", superimposed_img)
  # return Grad CAM
  return superimposed_img

def counterfactural_blacked_out(img, model_name, alpha):
  model_builder, img_size, preprocess_input, decode_predictions, last_conv_layer_name, target_size = architecture(model_name)

  #load image (np.array)
  img_np_ctfl = numpy.array(img)
  #ctcl img
  model = model_builder

  last_conv_layer = model.get_layer(last_conv_layer_name)
  last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

  classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
  x = classifier_input
  for layer_name in ["avg_pool", "predictions"]:
      x = model.get_layer(layer_name)(x)
  classifier_model = tf.keras.Model(classifier_input, x)

  with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(img_np_ctfl[np.newaxis, ...])
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]
  grads = tape.gradient(top_class_channel, last_conv_layer_output)
  pooled_grads = tf.reduce_mean(-1 * grads, axis=(0, 1, 2))
  last_conv_layer_output = last_conv_layer_output.numpy()[0]
  pooled_grads = pooled_grads.numpy()
  for i in range(pooled_grads.shape[-1]):
      last_conv_layer_output[:, :, i] *= pooled_grads[i]
  
  # Average over all the filters to get a single 2D array
  ctfcl_gradcam = np.mean(last_conv_layer_output, axis=-1)
  # Normalise the values
  ctfcl_gradcam = np.clip(ctfcl_gradcam, 0, np.max(ctfcl_gradcam)) / np.max(ctfcl_gradcam)
  if model_name == "xception":
    ctfcl_gradcam = cv2.resize(ctfcl_gradcam, (299, 299))
    mask = cv2.resize(ctfcl_gradcam, (299, 299))
  else:
    ctfcl_gradcam = cv2.resize(ctfcl_gradcam, (224, 224))

  #blacked_out
    mask = cv2.resize(ctfcl_gradcam, (224, 224))
  mask[mask > 0.1] = 255
  mask[mask != 255] = 0
  mask = mask.astype(bool)
  ctfctl_image = img.copy()
  ctfctl_image[mask] = (0, 0, 0)
  #print("ctfcl image", ctfctl_image)
  #print("image", img)
  blacked = ctfctl_image * alpha + img
  blacked = array_to_img(blacked)


  return blacked

def image_to_base64(image):
    #img = Image.open(image_path)
    output_buffer = BytesIO()
    image.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str

def pil_to_b64(im, enc_format="png", **kwargs):
    """
    Converts a PIL Image into base64 string for HTML displaying
    :param im: PIL Image object
    :param enc_format: The image format for displaying. If saved the image will have that extension.
    :return: base64 encoding
    """

    buff = BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded

    
def test(model_name = 'resnet50', filename = 'assets/katze.JPG', top = 10, alpha = 0.9):
  #1 select model
  model_builder, img_size, preprocess_input, decode_predictions, last_conv_layer_name, target_size = architecture(model_name)
  #2 load image
  #print(filename, target_size)
  img = load_image(filename, target_size)
  #3 show
  test_show(img)

  #4 predict image
  df_prediction = predict_img(img, model_builder, preprocess_input, decode_predictions)
  print(df_prediction)
  #3 make gradcam heatmap
  # Generate class activation heatmap

  # Remove last layer's softmax  
  # model_builder.layers[-1].activation = None

  heatmap = make_gradcam_heatmap(img, model_builder, last_conv_layer_name)
  test_show(heatmap)

  #4
  gradcam_img = save_and_display_gradcam(img, heatmap, alpha = alpha)
  test_show(gradcam_img)
 
#test()