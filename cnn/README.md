# CNN models

This directory comprises of CNN Keras models for 
  - [Adam Optimizer](./blerssi-cnn-adam.py)
  - [SGD Optimizer](./blerssi-cnn-sgd.py)
  
 These python models if run directly will take default hyperparameters, 
 e.g. *learning rate - 0.001, beta1 - 0.9 for Adam optimizer*
 
 To try custom hyperparameter values to the model with Adam Optimizer, one needs to
 ```
 python3 blerssi-cnn-adam.py --learning_rate=<> --beta1=<>
 ```
 
 To try custom hyperparameter values to the model with SGD Optimizer, one needs to
 ```
 python3 blerssi-cnn-sgd.py --learning_rate=<> --momentum=<>
 ```
 
 It also has reference [Dockerfile](./Dockerfile) for packaging the models into a docker image.
