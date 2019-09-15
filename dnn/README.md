# DNN Models

This directory comprises of DNN Keras models for 
  - [Adam Optimizer](./blerssi-dnn-adam.py)
  - [SGD Optimizer](./blerssi-dnn-sgd.py)
  
 These python models if run directly will take default hyperparameters, 
 e.g. *learning rate - 0.001, beta1 - 0.9 for Adam optimizer*
 
 To try custom hyperparameter values to the model with Adam Optimizer, one needs to
 ```
 python3 blerssi-dnn-adam.py --learning_rate=<> --beta1=<>
 ```
 
 To try custom hyperparameter values to the model with SGD Optimizer, one needs to
 ```
 python3 blerssi-dnn-sgd.py --learning_rate=<> --momentum=<>
 ```
 
 This directory also comprises of DNN Keras models to train on augmented & public datasets 
  - [Model using std. autoencoder augmentation](./blerssi-dnn-aug-stdenc.py)
  - [Model using naive augmentation](./blerssi-dnn-aug-reg.py)
  - [Model using hybrid augmentation](./blerssi-dnn-aug-hyb.py)
  
 
 It also has a reference [Dockerfile](./Dockerfile) for packaging the models into a docker image.
