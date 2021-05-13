# the imports

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
from keras.optimizers import SGD
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Flatten
from tensorflow.keras.models import Model
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


# Paths of the base and style image.
base_image_path = ""
style_image_path = ""


# gram matrix function
# was well explained it here: https://www.youtube.com/watch?v=DEK-W5cxG-g

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


# find the cost function of the styles within the layers

def cost_style(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))



# cost function b/w the content image and the generated image

def cost_content(base, combination):
    return tf.reduce_sum(tf.square(combination - base))



# Get the pretrained model.

model = vgg19.VGG19(weights="imagenet", include_top=False)

model.summary()



outputs_dict= dict([(layer.name, layer.output) for layer in model.layers])

feature_extractor = Model(inputs=model.inputs, outputs=outputs_dict)


layer_style = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

layer_content = "block5_conv2"

content_weight = 2.5e-8
style_weight = 1e-6




# for calculating the total loss function, adding both the losses.

def loss_function(combination_image, base_image, style_reference_image):

    # 1. Combine all the images in the same tensioner.
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )

    # 2. Get the values in all the layers for the three images.
    features = feature_extractor(input_tensor)

    #3. Inicializar the loss

    loss = tf.zeros(shape=())

    # 4. Extract the content layers + content loss
    layer_features = features[layer_content]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]

    loss = loss + content_weight * cost_content(
        base_image_features, combination_features
    )
    # 5. Extraer the style layers + style loss
    for layer_name in layer_style:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = cost_style(style_reference_features, combination_features)
        loss += (style_weight / len(layer_style)) * sl

    return loss



# computing the gradient descent

@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = loss_function(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads


def preprocess_image(image_path):
    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)



def deprocess_image(x):

    # Convertimos el tensor en Array
    x = x.reshape((img_nrows, img_ncols, 3))

    # Hacemos que no tengan promedio 0
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # Convertimos de BGR a RGB.
    x = x[:, :, ::-1]

    # Nos aseguramos que están entre 0 y 255
    x = np.clip(x, 0, 255).astype("uint8")

    return x



def result_saver(iteration):
  # Create name
    now = datetime.now()
    now = now.strftime("get_ipython().run_line_magic("Y%m%d_%H%M%S")", "")
    # Save image
    img = deprocess_image(combination_image.numpy())
    keras.preprocessing.image.save_img(f"{iteration}.png", img)



width, height = keras.preprocessing.image.load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

optimizer = SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    )
)

base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_image_path)
combination_image = tf.Variable(preprocess_image(base_image_path))

iterations = 4000

for i in tqdm(range(1, iterations + 1)):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 10 == 0:
        print("Iteration get_ipython().run_line_magic("d:", " loss=%.2f\" % (i, loss))")
        result_saver(i)







