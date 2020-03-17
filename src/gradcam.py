import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model

def get_grad_cam(image, model, layer_name, class_index):
    """tf2.0 implementation of gradient weighted class activation map 
    Arguments:
      image: Input image, processed input image with batch dimension, shape=(1, H, W, C)
      model: CNN model
      layer_name: relu of the last convolution layer in the network
      class index: index of the class for which Grad CAM needs to be calculated
    Returns: Class decriminative heatmap for class_index class for given image"""
    
    #create model with ouput as last convolution layer and model output
    gradCamModel = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])
        
    #record operations for auto diff
    with tf.GradientTape() as tape:
        inputs = tf.cast(image, tf.float32)
        [convOut, predictions] = gradCamModel(inputs)
        loss = predictions[:, class_index]
        
    grads = tape.gradient(loss, convOut)
    castConvOut = tf.cast(convOut>0, tf.float32)
    castGrads = tf.cast(grads>0,tf.float32)
    guidedGrads = castConvOut*castGrads*grads
    
    convOut = convOut[0]
    guidedGrads = guidedGrads[0]
    
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOut), axis=-1)
    h, w = image.shape[1:3]
    heatmap = cv2.resize(cam.numpy(), (w, h))
    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + 1e-8
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")
    return heatmap

def overlay_heatmap(image, heatmap):
    """Function applies colormap to heatmap and overlays the colored heatmap on the input image"""
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET).astype('float32')
    output = cv2.addWeighted(image, 0.5, colored_heatmap, 0.5, 0)
    numer = output - np.min(output)
    denom = (output.max() - output.min()) + 1e-8
    output = numer / denom
    output = (output * 255).astype("uint8")
    return output