from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import urllib.request

from torchvision import transforms
import tensorflow as tf
import numpy as np
from PIL import Image

from aws_textract.e2e_wrapper import single_file_show
from aws_textract.e2e_wrapper import  predict_single_path

img_width = 200
img_height = 50
batch_size = 16

characters = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ']


class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


urllib.request.urlretrieve('https://github.com/meliiwamd/OpticalCharacterRecognition/raw/main/ocr.h5', '.\\ocr.h5')
model = tf.keras.models.load_model('.\\ocr.h5', custom_objects={"CTCLayer": CTCLayer})

char_to_num = tf.keras.layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)

# Mapping Int Back To Original Char
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def EncodeSingleSample(img_path, label):
    print('gsdafwds')
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=3)

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])

    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

    return {"image": img, "label": label}


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
              :, :16
              ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def prediction(images):
    print("Number of images found: ", len(images), '\n')
    result = []

    test_dataset = tf.data.Dataset.from_tensor_slices((images, ['0' * 16] * len(images)))
    test_dataset = (
        test_dataset.map(
            EncodeSingleSample, num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    print(1)
    prediction_model = tf.keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )

    print(1)
    for batch in test_dataset.take(1):
        batch_images = batch["image"]
        batch_labels = batch["label"]

        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)

        result = result + pred_texts

    print(1)
    return result


@api_view(['POST', ])
def get_textract(request):
    image = request.data['image']
    length = len(str(image)) - 4
    print(length)
    try:
        img = Image.open(image)
        img.save("./geeks.jpg")

        pred_class, pred_label = predict_single_path("./geeks.jpg", '0' * length)
        return Response({"pred_class": pred_class, "pred_label": pred_label}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"result": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST', ])
def get_textract2(request):
    image = request.data['image']
    try:
        img = Image.open(image)
        img.save("./geeks.jpg")
        images = ["./geeks.jpg"]
        results = prediction(images)
        return Response({"result": results}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"result": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
