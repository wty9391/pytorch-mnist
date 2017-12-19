import numpy
from flask import Flask, jsonify, render_template, request
from PIL import Image

import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

import LeNet
# webapp
app = Flask(__name__)

def predict_with_pretrain_model(sample,model):
    """
    Args:
        sample: A integer ndarray indicating an image, whose shape is (28,28).
    
    Returns:
        A list consists of 10 double numbers, which denotes the probabilities of numbers(from 0 to 9).
        like [0.1,0.1,0.2,0.05,0.05,0.1,0.1,0.1,0.1,0.1].
    """
    #In MNIST dataset pixel values are 0 to 255.
    #0 means background (white), 255 means foreground (black). 
    #In our dataset, 0 means black, 255 means white
    sample = -sample + 255
    img = Image.fromarray(sample)
    transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
    sample = transform(img).float()
    sample = Variable(sample, volatile=True)
    sample = sample.unsqueeze(0)
    out = model.predict(sample)
    return out.data[0].tolist()

@app.route('/api/mnist', methods=['POST'])
def mnist():
    model = LeNet.Net()
    model.load_state_dict(torch.load('./results/model.pt'))
    input = ((numpy.array(request.json, dtype=numpy.uint8))).reshape(28, 28)
    output = predict_with_pretrain_model(input,model)
    return jsonify(results=output)


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
