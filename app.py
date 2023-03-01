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
# from emo_distribution import AffectiveFace
from scipy import stats
import gc

app = Flask(__name__)
CORS(app)
model = saved_model.load('./ModelSaved')
# prior = AffectiveFace()
gmm_h = stats.multivariate_normal(mean=[1.574, -0.990, -0.130],
                                  cov=[[0.968, 0.012, -0.025],
                                       [0.012, 0.987, -0.083],
                                       [-0.025, -0.083, 0.963]])

gmm_s1 = stats.multivariate_normal(mean=[-1.081, 0.756, 0.673],
                                   cov=[[1.151, -0.019, -0.053],
                                        [-0.019, 1.010, 0.054],
                                        [-0.053, 0.054, 0.864]])

gmm_s2 = stats.multivariate_normal(mean=[-0.628, 0.526, 0.485],
                                   cov=[[0.768, -0.028, 0.006],
                                        [-0.028, 0.764, -0.052],
                                        [0.006, -0.052, 0.832]])


def _numpy_to_base64(image_np):
    data = cv2.imencode('.jpg', image_np)[1]
    image_bytes = data.tobytes()
    image_base4 = base64.b64encode(image_bytes).decode('utf8')
    return image_base4


def gmm_density(locs, emotion_id):
    if emotion_id == 1:  # happy
        density = gmm_h.pdf(locs)
    elif emotion_id == 2:
        density = gmm_s1.pdf(locs) * 0.582 + gmm_s2.pdf(locs) * 0.418

    return density


@app.route("/", methods=['GET', 'POST'])
def img():

    loc = json.loads(request.get_data())
    prediction = model(loc['data'])
    img_array_l = prediction[0].numpy()[0, :, :, :].squeeze() * 255
    img_base4_l = _numpy_to_base64(img_array_l)
    img_array_r = prediction[0].numpy()[1, :, :, :].squeeze() * 255
    img_base4_r = _numpy_to_base64(img_array_r)

    del loc, prediction, img_array_l, img_array_r
    gc.collect()

    return {'left': 'data:image/png;base64,' + str(img_base4_l),
            'right': 'data:image/png;base64,' + str(img_base4_r)}
    # return render_template('test.html')


@app.route("/fx/<int:emotion_id>", methods=['GET', 'POST'])
def fx(emotion_id):  # emotion_id is in range 0-6

    loc = json.loads(request.get_data())
    # mix_p = prior.get_density(loc['data'], emotion_id)
    model_density = gmm_density(loc['data'], emotion_id)

    return {'density': model_density.tolist()}


# flask run --host=0.0.0.0 --port=80
# curl -d '{"data": [[1,1,1], [0, 0, 0]]}' -X POST http://127.0.0.1:5000/fx/2
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv("PORT", 5000))


