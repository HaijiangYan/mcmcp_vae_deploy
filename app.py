#!/Yan/miniforge3/envs/
# -*- coding:utf-8 -*-


from flask import Flask, request
from flask_cors import CORS
# from flask import render_template
from tensorflow import saved_model
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
cov_mat = [[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]
gmm_h = stats.multivariate_normal(mean=[1.572, -0.994, -0.140],
                                  cov=cov_mat)

gmm_s = stats.multivariate_normal(mean=[-0.892, 0.665, 0.584],
                                  cov=cov_mat)

gmm_neutral = stats.multivariate_normal(mean=[-0.511, -0.965, 1.249],
                                        cov=cov_mat)

gmm_fear = stats.multivariate_normal(mean=[-0.655, -1.201, -1.109],
                                     cov=cov_mat)

gmm_dis = stats.multivariate_normal(mean=[0.936, 1.179, -0.947],
                                    cov=cov_mat)

gmm_ang = stats.multivariate_normal(mean=[0.635, 1.242, 1.347],
                                    cov=cov_mat)

gmm_sup = stats.multivariate_normal(mean=[-1.371, 0.742, -1.326],
                                    cov=cov_mat)


def _numpy_to_base64(image_np):
    data = cv2.imencode('.jpg', image_np)[1]
    image_bytes = data.tobytes()
    image_base4 = base64.b64encode(image_bytes).decode('utf8')
    return image_base4


def gmm_density(locs, emotion_id):
    if emotion_id == 1:  # happy
        density = gmm_h.pdf(locs) - (gmm_s.pdf(locs) + gmm_neutral.pdf(locs) + gmm_fear.pdf(locs)
                                     + gmm_dis.pdf(locs) + gmm_ang.pdf(locs) + gmm_sup.pdf(locs)) / 6
    elif emotion_id == 2:
        density = gmm_s.pdf(locs) - (gmm_h.pdf(locs) + gmm_neutral.pdf(locs) + gmm_fear.pdf(locs)
                                     + gmm_dis.pdf(locs) + gmm_ang.pdf(locs) + gmm_sup.pdf(locs)) / 6

    if density[0] < 0:
        density[0] = 0

    if density[1] < 0:
        density[1] = 0

    return density


@app.route("/", methods=['GET', 'POST'])
def img():
    loc = json.loads(request.get_data())
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

    loc = json.loads(request.get_data())
    # mix_p = prior.get_density(loc['data'], emotion_id)
    model_density = gmm_density(loc['data'], emotion_id)

    return {'density': model_density.tolist()}


# flask run --host=0.0.0.0 --port=80
# curl -d '{"data": [[1,1,1], [0, 0, 0]]}' -X POST http://127.0.0.1:5000/fx/2
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv("PORT", 5000))
# host='0.0.0.0', port=os.getenv("PORT", 5000)
