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

app = Flask(__name__)
CORS(app)
model = saved_model.load('./ModelSaved')
# prior = AffectiveFace()
gmm_h1 = stats.multivariate_normal(mean=[1.38445539, -0.86085914, -0.12753441],
                                   cov=[[0.14541639, 0.01199723, -0.01853188],
                                        [0.01199723, 0.04749794, -0.01231523],
                                        [-0.01853188, -0.01231523, 0.0352132]])

gmm_h2 = stats.multivariate_normal(mean=[1.76839786, -1.0224419, -0.47976337],
                                   cov=[[0.20442699, 0.20461399, -0.1039588],
                                        [0.20461399, 0.30873015, -0.17221446],
                                        [-0.1039588, -0.17221446, 0.21401007]])

gmm_h3 = stats.multivariate_normal(mean=[1.7216995, -1.21029812, 0.18016213],
                                   cov=[[0.02889061, -0.03503325, 0.04589235],
                                        [-0.03503325, 0.25920628, -0.06107783],
                                        [0.04589235, -0.06107783, 0.12520434]])

gmm_s1 = stats.multivariate_normal(mean=[-0.69823855, 0.53385238, 0.55744477],
                                   cov=[[0.08809386, -0.02021937, -0.02022361],
                                        [-0.02021937, 0.0324342, 0.00799594],
                                        [-0.02022361, 0.00799594, 0.044204]])

gmm_s2 = stats.multivariate_normal(mean=[-1.37001812, 0.98972423, 0.649176],
                                   cov=[[0.32913519, 0.08802234, -0.07693232],
                                        [0.08802234, 0.19912714, 0.03047114],
                                        [-0.07693232, 0.03047114, 0.08038485]])


def _numpy_to_base64(image_np):
    data = cv2.imencode('.jpg', image_np)[1]
    image_bytes = data.tobytes()
    image_base4 = base64.b64encode(image_bytes).decode('utf8')
    return image_base4


def gmm_density(locs, emotion_id):
    if emotion_id == 1:  # happy
        density = gmm_h1.pdf(locs) * 0.47877633\
                  + gmm_h2.pdf(locs) * 0.26221939\
                  + gmm_h3.pdf(locs) * 0.25900429
    elif emotion_id == 2:
        density = gmm_s1.pdf(locs) * 0.71121901 \
                  + gmm_s2.pdf(locs) * 0.28878099

    return density


@app.route("/", methods=['GET', 'POST'])
def img():
    loc = request.get_data()
    loc = json.loads(loc)
    prediction = model(loc['data'])
    img_array_l = prediction[0].numpy()[0, :, :, :].squeeze() * 255
    img_base4_l = _numpy_to_base64(img_array_l)
    img_array_r = prediction[0].numpy()[1, :, :, :].squeeze() * 255
    img_base4_r = _numpy_to_base64(img_array_r)
    return {'left': 'data:image/png;base64,' + str(img_base4_l),
            'right': 'data:image/png;base64,' + str(img_base4_r)}
    # return render_template('test.html')


@app.route("/fx/<int:emotion_id>", methods=['GET', 'POST'])
def fx(emotion_id):  # emotion_id is in range 0-6
    loc = request.get_data()
    loc = json.loads(loc)
    # mix_p = prior.get_density(loc['data'], emotion_id)
    model_density = gmm_density(loc['data'], emotion_id)

    return {'density': model_density.tolist()}


# flask run --host=0.0.0.0 --port=80
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv("PORT", 5000))


