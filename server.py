import os
from flask import Flask

app = Flask(__name__)

def predict_core(filepath):
    import caffe
    import sys
    import time

    def resize_mean(mean):
        m_min, m_max = mean.min(), mean.max()
        normal_mean = (mean - m_min) / (m_max - m_min)
        return caffe.io.resize_image(normal_mean.transpose((1,2,0)), [256, 256]).transpose((2,0,1)) * (m_max - m_min) + m_min

    caffe.set_mode_cpu()
    mean_filename='/root/work_dir/train/mean.binaryproto'
    proto_data = open(mean_filename, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean = caffe.io.blobproto_to_array(a)[0]
    mean = resize_mean(mean)
    age_net = caffe.Classifier('/root/agegenderdeeplearning/car_net_definition/deploy.prototxt', '/root/results/caffenet_car_train_iter_34000.caffemodel', mean=mean, channel_swap=(2,1,0), raw_scale=255, image_dims=(256, 256))
    img = caffe.io.load_image(filepath)
    scores = age_net.predict([img])
    return scores[0]

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filepath = os.path.join(os.path.dirname(__file__), 'upload', file.filename)
    file.save(filepath)
    data = dict(scores=predict_core(filepath))
    return flask.jsonify(**data)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
