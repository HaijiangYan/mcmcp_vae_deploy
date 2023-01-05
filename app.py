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

app = Flask(__name__)
CORS(app)
model = saved_model.load('./model')


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


@app.route("/category", methods=['GET', 'POST'])
def cat():
    loc = request.get_data()
    loc = json.loads(loc)
    prediction = model.decoder(loc['data'])
    category = prediction[1].numpy()
    return repr(json.dumps({'cat': int(np.argmax(category, axis=1)), 'value': float(np.max(category, axis=1))}))


# flask run --host=0.0.0.0 --port=80
# docker build -t cvae_mcmcp .
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv("PORT", 5000))
# docker buildx build --load --platform linux/amd64 -t registry.heroku.com/mcmcp/web .
# docker push registry.heroku.com/mcmcp/web:latest  
# heroku container:release web -a mcmcp 

