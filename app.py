#!/Yan/miniforge3/envs/
# -*- coding:utf-8 -*-


from flask import Flask, request
from flask_cors import CORS
# from flask import render_template
from tensorflow import saved_model
import numpy as np
import os
import json
import cv2
import base64
import emo_distribution

app = Flask(__name__)
CORS(app)
model = saved_model.load('./ModelSaved')


def numpy_to_base64(image_np):
    data = cv2.imencode('.jpg', image_np)[1]
    image_bytes = data.tobytes()
    image_base4 = base64.b64encode(image_bytes).decode('utf8')
    return image_base4


@app.route("/", methods=['GET', 'POST'])
def img():
    loc = request.get_data()
    loc = json.loads(loc)
    prediction = model.decoder(loc['data'])
    img_array = prediction[0].numpy()[0, :, :, :].squeeze() * 255
    img_base4 = numpy_to_base64(img_array)
    # im = Image.fromarray(img_array)
    # im = im.convert('L')
    # im.save('sample.png')
    return 'data:image/png;base64,' + str(img_base4)
    # return render_template('test.html')


@app.route("/fx/<int:emotion_id>", methods=['GET', 'POST'])
def fx(emotion_id):  # emotion_id is in range 0-6
    loc = request.get_data()
    loc = json.loads(loc)
    mix_p = emo_distribution.get_distribution(loc['data'], emotion_id)
    return json.dumps({'density': mix_p.tolist()})


# flask run --host=0.0.0.0 --port=80
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv("PORT", 5000))


