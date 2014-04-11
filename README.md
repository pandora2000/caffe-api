# Caffe API

An API interface to some [Caffe] models. We use Flask to expose Caffe models previously trained. 

We have used the Imagenet python wrapper from the Caffe repository and have implemented (based on that) a Lenet wrapper, in a local module `_caffe`. The image preprocess is done by the wrappers, but for the Lenet we must provide a centered image of a number, to match the structure of the images in the MNIST dataset used for training it.

## Usage

Run the script and wait for the models to load. After all models are loaded the server will be working on port 5000, the models being available in `/lenet` and `/imagenet`, respectively. You can POST a image to a model using, for example, the following command:

```
curl --form "image=@/path/to/image/image.jpg" [YOUR_IP]:5000/[MODEL_NAME]
```

The response is a JSON file containing the best class prediction, a list with the classes probabilities sorted by descending order and a list with the classes sorted by the propabiblities in descending order. Here is a response example:

```
{
  "best": 0, 
  "predictions": [
    0, 
    6, 
    9, 
    7, 
    8, 
    5, 
    2, 
    1, 
    3, 
    4
  ], 
  "probabilities": [
    0.9979206919670105, 
    0.0007992387982085347, 
    0.0005818080389872193, 
    0.00036282758810557425, 
    0.0001612447085790336, 
    0.00010736921103671193, 
    4.2144984035985544e-05, 
    1.3334250979823992e-05, 
    6.253382252907613e-06, 
    5.1289121074660216e-06
  ]
}
```

[Caffe]:https://github.com/BVLC/caffe
