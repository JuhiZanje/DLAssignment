import argparse

import numpy as np
import wandb
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

# from Classes import MultiLayerPerceptron
# from Classes import LossActFunc

class MultiLayerPerceptron():

  def __init__(
        self,
        train_data,
        train_labels,
        val_data,
        val_labels,
        layer_sizes,
        initialization,
        loss_act_func,
        batch_size=4,
        epochs=1,
        learning_rate=0.1,
        weight_decay=0,
        optimizer="sgd",
        momentum=0.5,
        beta=0.5,
        beta1=0.5,
        beta2=0.5,
        epsilon=0.000001,
        printFlag="print"
        ):

      self.parameters = {}
      self.gradients = {}
      self.train_data = train_data
      self.train_labels = train_labels
      self.val_data=val_data
      self.val_labels=val_labels
      self.no_of_samples = len(train_data)
      self.no_of_features = train_data.shape[1] #size of column of train_data
      self.no_of_classes = 10 #size of column of train_labels
      self.layer_sizes = layer_sizes
      self.batch_size=batch_size
      # self.act_func=act_func
      self.epochs=epochs
      self.learning_rate=learning_rate
      self.weight_decay=weight_decay
      self.optimizer=optimizer
      self.loss_act_func=loss_act_func
      self.initialization = initialization
      self.momentum=momentum
      self.beta=beta
      self.beta1=beta1
      self.beta2=beta2
      self.epsilon=epsilon
      self.printFlag=printFlag

      self.A = {}
      self.H = {}

#function to initialize Weights either using random initializer or Xavior
  def init_weights(self):
    if self.initialization=="random":
      # print("init random")
      for i in range(len(self.layer_sizes) - 1):
          #input layer(n) and 1st hidden layer(m) will have weight vector w1->n*m and bias b1->1*m
          #init W to random values
          self.parameters[f'w_{i+1}'] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1])
          #init B to 0
          self.parameters[f'b_{i+1}'] = np.random.randn(1,self.layer_sizes[i+1])

    if self.initialization=="Xavier":
      print("inint Xavier")
      for i in range(len(self.layer_sizes) - 1):
        x_fact = np.sqrt(6 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
        self.parameters[f'w_{i+1}'] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * x_fact
        self.parameters[f'b_{i+1}'] = np.random.randn(1,self.layer_sizes[i+1]) * x_fact

#softmax fynction
  def softmax(self, x):
    max_x = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_x)
    return exps / np.sum(exps, axis=1, keepdims=True)

#derivation of softmax function
  def softmax_d(self,x):
    softmax_x = self.softmax(x)
    return softmax_x*(1-softmax_x)

#feed forward Function
  def feed_forwards(self, input_data):      
    for i in range(len(self.layer_sizes) - 1): 
        # if(i==0) :#input is our images
        if(i==0):
          preactivation=np.dot(input_data , self.parameters[f'w_{i+1}']) + self.parameters[f'b_{i+1}']
        else:
          preactivation = np.dot(self.H[i], self.parameters[f'w_{i+1}']) + self.parameters[f'b_{i+1}']
        self.A[i+1]=preactivation

        if i < len(self.layer_sizes) - 2 : 
          #if we havent reached the last layer
          #give preactivation output of previous layer to the acivation function                        
            activation = self.loss_act_func.activation_func(self.A[i+1],"real")
        else :
            #if we reached last layer then use softmax as activation  
            activation=self.softmax(self.A[i+1])
        self.H[i+1]=activation

    return self.H[len(self.layer_sizes)-1]

#back propogation method
  def back_prop(self, input_data, true_label,pred_label):
    #check which loss function we are using and change dL-dA according to it 
    if ((self.loss_act_func).get_name()=="cross_entropy"):
      # print("cross_entropy")
      dL_dA = pred_label - true_label
    elif ((self.loss_act_func).get_name()=="mean_squared_error"):
      dL_dA = ((pred_label - true_label) * self.softmax_d(self.A[len(self.layer_sizes)-1])) / true_label.shape[0]

    #implementing chain rule
    for i in range(len(self.layer_sizes) - 1, 1, -1): # i->4 3 2

        self.gradients[f'w_{i}']=np.dot(self.H[i-1].T, dL_dA)
        self.gradients[f'b_{i}']=np.sum(dL_dA , axis=0,keepdims=True)

        dL_dH=np.dot(dL_dA, (self.parameters[f'w_{i}']).T )
        dL_dA = dL_dH * self.loss_act_func.activation_func(self.A[i-1],"derivative")

    self.gradients[f'w_{1}']=np.dot(input_data.T, dL_dA )
    self.gradients[f'b_{1}']=np.sum(dL_dA , axis=0,keepdims=True)

#All optimizers are implemented here
  def optimizer_func(self):

      if(self.optimizer=="sgd"):
          #Implementing SGD
          for i in range(self.epochs):
              #j loop will go for all the batches created
              for j in range(0,self.no_of_samples,self.batch_size):
                  start = j
                  end = start + self.batch_size
                  Y_train_labels = self.train_labels[start:end]
                  X_train_images = self.train_data[start:end]

                  pred_labels=self.feed_forwards(X_train_images)
                  self.back_prop(X_train_images, Y_train_labels,pred_labels)

                  #update weights and biases
                  for k in range(len(self.layer_sizes) - 1):# 0->3
                      self.gradients[f'w_{k+1}']=self.gradients[f'w_{k+1}']/self.batch_size
                      self.gradients[f'b_{k+1}']=self.gradients[f'b_{k+1}']/self.batch_size

                      self.gradients[f'w_{k+1}'] =(self.gradients[f'w_{k+1}']) + self.weight_decay * self.parameters[f'w_{k+1}']
                      self.parameters[f'w_{k+1}'] -= self.learning_rate * (self.gradients[f'w_{k+1}'] )
                      self.parameters[f'b_{k+1}'] -= self.learning_rate * (self.gradients[f'b_{k+1}'] )

              #finding training/validation loss/accuracy
              pred_labels = self.feed_forwards(self.train_data)
              tra_acc = self.accuracy(pred_labels,self.train_labels)
              tra_loss = self.loss_act_func.loss_functions(pred_labels, self.train_labels)

              pred_labels = self.feed_forwards(self.val_data)
              val_acc = self.accuracy(pred_labels,self.val_labels)
              val_loss = self.loss_act_func.loss_functions(pred_labels, self.val_labels)

              if(self.printFlag=="print"):
                print(f"Epoch {i+1}, Training Loss: {tra_loss} , Training Accuracy: {tra_acc} , Validation Loss: {val_loss} Validation Accuracy: {val_acc}")
              else:
                wandb.log({"Epoch": i+1, "Training_Loss": tra_loss, "Training_Accuracy": tra_acc,"Validation_Loss": val_loss,"Validation_Accuracy": val_acc})

      if(self.optimizer=="momentum"):
        #init velocities for w and b to zeros        
        velocities={}
        for i in range(len(self.layer_sizes) - 1):
          velocities[f'w_{i+1}'] = np.zeros((self.layer_sizes[i], self.layer_sizes[i+1]))
          velocities[f'b_{i+1}'] = np.zeros((1, self.layer_sizes[i+1]))

        for i in range(self.epochs):
          for j in range(0,self.no_of_samples,self.batch_size):
            start = j
            end = start + self.batch_size
            Y_train_labels = self.train_labels[start:end]
            X_train_images = self.train_data[start:end]

            pred_labels=self.feed_forwards(X_train_images)
            self.back_prop(X_train_images, Y_train_labels,pred_labels)
            #update weights and bias as per momentum updation rule
            for k in range(len(self.layer_sizes) - 1):
                self.gradients[f'w_{k+1}']=self.gradients[f'w_{k+1}']/self.batch_size
                self.gradients[f'b_{k+1}']=self.gradients[f'b_{k+1}']/self.batch_size

                self.gradients[f'w_{k+1}'] =(self.gradients[f'w_{k+1}']) + self.weight_decay * self.parameters[f'w_{k+1}']
                # Update velocities with momentum
                velocities[f'w_{k+1}'] = self.momentum * velocities[f'w_{k+1}'] + self.learning_rate * (self.gradients[f'w_{k+1}'])
                velocities[f'b_{k+1}'] = self.momentum * velocities[f'b_{k+1}'] + self.learning_rate * (self.gradients[f'b_{k+1}'])

                # Update parameters with momentum
                self.parameters[f'w_{k+1}'] -= velocities[f'w_{k+1}']
                self.parameters[f'b_{k+1}'] -= velocities[f'b_{k+1}']
          
          #train/validation data loss/accuracy
          pred_labels = self.feed_forwards(self.train_data)
          tra_acc = self.accuracy(pred_labels,self.train_labels)
          tra_loss = self.loss_act_func.loss_functions(pred_labels, self.train_labels)

          pred_labels = self.feed_forwards(self.val_data)
          val_acc = self.accuracy(pred_labels,self.val_labels)
          val_loss = self.loss_act_func.loss_functions(pred_labels, self.val_labels)
          if(self.printFlag=="print"):
            print(f"Epoch {i+1}, Training Loss: {tra_loss} , Training Accuracy: {tra_acc} , Validation Loss: {val_loss} Validation Accuracy: {val_acc}")
          else:
            wandb.log({"Epoch": i+1, "Training_Loss": tra_loss, "Training_Accuracy": tra_acc,"Validation_Loss": val_loss,"Validation_Accuracy": val_acc})

      if(self.optimizer=="nag"):
          '''
          Epoch 100, Loss: 0.8561739933889985 Acc: 63.355
          TEST LOSS: 0.8864 ACCURACY: 62.7400

          Epoch 20, Loss: 1.013375573148102 Acc: 61.795
          TEST LOSS: 1.0499 ACCURACY: 60.2300
          '''
          # Initialize velocities for weights and biases
          velocities={}
          for i in range(len(self.layer_sizes) - 1):
              velocities[f'w_{i+1}'] = np.zeros((self.layer_sizes[i], self.layer_sizes[i+1]))
              velocities[f'b_{i+1}'] = np.zeros((1, self.layer_sizes[i+1]))

          for i in range(self.epochs):
              for j in range(0,self.no_of_samples,self.batch_size):
                  start = j
                  end = start + self.batch_size
                  Y_train_labels = self.train_labels[start:end]
                  X_train_images = self.train_data[start:end]

                  '''
                  storing old weights in self.parameters[f'w_{k+1}_orig']
                  so that i can use it later and
                  updating self.parameters[f'w_{k+1}'] so that same feed forward function can be used.
                  '''
                  for k in range(len(self.layer_sizes) - 1):
                      self.parameters[f'w_{k+1}_orig']=self.parameters[f'w_{k+1}']
                      self.parameters[f'b_{k+1}_orig']=self.parameters[f'b_{k+1}']

                      self.parameters[f'w_{k+1}'] = self.parameters[f'w_{k+1}'] - self.momentum * velocities[f'w_{k+1}']
                      self.parameters[f'b_{k+1}'] = self.parameters[f'b_{k+1}'] - self.momentum * velocities[f'b_{k+1}']

                  #feed forward and backprop is applied after changing weights and biases
                  pred_labels=self.feed_forwards(X_train_images)
                  self.back_prop(X_train_images, Y_train_labels,pred_labels)

                  '''
                  here 1st we will set original weights to self.parameters[f'w_{k+1}']
                  and then update weigths and biases.
                  '''
                  for k in range(len(self.layer_sizes) - 1):

                      self.gradients[f'w_{k+1}']=self.gradients[f'w_{k+1}']/self.batch_size
                      self.gradients[f'b_{k+1}']=self.gradients[f'b_{k+1}']/self.batch_size

                      self.gradients[f'w_{k+1}'] =(self.gradients[f'w_{k+1}']) + self.weight_decay * self.parameters[f'w_{k+1}']

                      velocities[f'w_{k+1}'] = self.momentum * velocities[f'w_{k+1}'] + self.learning_rate * (self.gradients[f'w_{k+1}'] )
                      velocities[f'b_{k+1}'] = self.momentum * velocities[f'b_{k+1}'] + self.learning_rate * (self.gradients[f'b_{k+1}'] )

                      self.parameters[f'w_{k+1}']=self.parameters[f'w_{k+1}_orig']
                      self.parameters[f'b_{k+1}']=self.parameters[f'b_{k+1}_orig']

                      self.parameters[f'w_{k+1}'] -= velocities[f'w_{k+1}']
                      self.parameters[f'b_{k+1}'] -= velocities[f'b_{k+1}']

              pred_labels = self.feed_forwards(self.train_data)
              tra_acc = self.accuracy(pred_labels,self.train_labels)
              tra_loss = self.loss_act_func.loss_functions(pred_labels, self.train_labels)

              pred_labels = self.feed_forwards(self.val_data)
              val_acc = self.accuracy(pred_labels,self.val_labels)
              val_loss = self.loss_act_func.loss_functions(pred_labels, self.val_labels)
              if(self.printFlag=="print"):
                print(f"Epoch {i+1}, Training Loss: {tra_loss} , Training Accuracy: {tra_acc} , Validation Loss: {val_loss} Validation Accuracy: {val_acc}")
              else:
                wandb.log({"Epoch": i+1, "Training_Loss": tra_loss, "Training_Accuracy": tra_acc,"Validation_Loss": val_loss,"Validation_Accuracy": val_acc})

      if(self.optimizer=="rmsprop"):
        '''
        Epoch 10, Loss: 0.6455246819185498 Acc: 75.41499999999999
        TEST LOSS: 0.6710 ACCURACY: 74.4000
        '''
        squared_gradients={}
        for i in range(len(self.layer_sizes) - 1):
            squared_gradients[f'w_{i+1}'] = np.zeros((self.layer_sizes[i], self.layer_sizes[i+1]))
            squared_gradients[f'b_{i+1}'] = np.zeros((1, self.layer_sizes[i+1]))

        for i in range(self.epochs):
            for j in range(0,self.no_of_samples,self.batch_size):
                start = j
                end = start + self.batch_size
                Y_train_labels = self.train_labels[start:end]
                X_train_images = self.train_data[start:end]

                pred_labels=self.feed_forwards(X_train_images)
                self.back_prop(X_train_images, Y_train_labels,pred_labels)

                for k in range(len(self.layer_sizes) - 1):

                    self.gradients[f'w_{k+1}']=self.gradients[f'w_{k+1}']/self.batch_size
                    self.gradients[f'b_{k+1}']=self.gradients[f'b_{k+1}']/self.batch_size

                    self.gradients[f'w_{k+1}'] =(self.gradients[f'w_{k+1}']) + self.weight_decay * self.parameters[f'w_{k+1}']

                    # Calculate the squared gradients
                    squared_gradients[f'w_{k+1}'] = self.beta * squared_gradients[f'w_{k+1}'] + (1 - self.beta) * ((self.gradients[f'w_{k+1}']) ** 2)
                    squared_gradients[f'b_{k+1}'] = self.beta * squared_gradients[f'b_{k+1}'] + (1 - self.beta) * ((self.gradients[f'b_{k+1}']) ** 2)

                    # Update parameters using RMSprop
                    self.parameters[f'w_{k+1}'] -= (self.learning_rate * self.gradients[f'w_{k+1}'] / (np.sqrt(squared_gradients[f'w_{k+1}']) + self.epsilon))
                    self.parameters[f'b_{k+1}'] -= (self.learning_rate * self.gradients[f'b_{k+1}'] / (np.sqrt(squared_gradients[f'b_{k+1}']) + self.epsilon))

            pred_labels = self.feed_forwards(self.train_data)
            tra_acc = self.accuracy(pred_labels,self.train_labels)
            tra_loss = self.loss_act_func.loss_functions(pred_labels, self.train_labels)

            pred_labels = self.feed_forwards(self.val_data)
            val_acc = self.accuracy(pred_labels,self.val_labels)
            val_loss = self.loss_act_func.loss_functions(pred_labels, self.val_labels)
            if(self.printFlag=="print"):
              print(f"Epoch {i+1}, Training Loss: {tra_loss} , Training Accuracy: {tra_acc} , Validation Loss: {val_loss} Validation Accuracy: {val_acc}")
            else:
              wandb.log({"Epoch": i+1, "Training_Loss": tra_loss, "Training_Accuracy": tra_acc,"Validation_Loss": val_loss,"Validation_Accuracy": val_acc})

      if(self.optimizer=="adam"):
        '''
        Epoch 10, Loss: 0.6223362734184407 Acc: 77.985
        TEST LOSS: 0.6613 ACCURACY: 76.2700
        '''
        moments={}
        for i in range(len(self.layer_sizes) - 1):
            moments[f'm_w_{i+1}'] = np.zeros((self.layer_sizes[i], self.layer_sizes[i+1]))
            moments[f'm_b_{i+1}'] = np.zeros((1, self.layer_sizes[i+1]))

        velocities={}
        for i in range(len(self.layer_sizes) - 1):
            velocities[f'v_w_{i+1}'] = np.zeros((self.layer_sizes[i], self.layer_sizes[i+1]))
            velocities[f'v_b_{i+1}'] = np.zeros((1, self.layer_sizes[i+1]))

        # t=0
        for i in range(self.epochs):
          t=i+1
          for j in range(0,self.no_of_samples,self.batch_size):
              # t=t+1
              start = j
              end = start + self.batch_size
              Y_train_labels = self.train_labels[start:end]
              X_train_images = self.train_data[start:end]

              pred_labels=self.feed_forwards(X_train_images)
              self.back_prop(X_train_images, Y_train_labels,pred_labels)
              # t=1
              for k in range(len(self.layer_sizes) - 1):

                  self.gradients[f'w_{k+1}'] = self.gradients[f'w_{k+1}'] / self.batch_size
                  self.gradients[f'b_{k+1}']=self.gradients[f'b_{k+1}']/self.batch_size

                  self.gradients[f'w_{k+1}'] = self.gradients[f'w_{k+1}'] + self.weight_decay * self.parameters[f'w_{k+1}']

                  # Update moments and velocities with Adam
                  moments[f'm_w_{k+1}'] = self.beta1 * moments[f'm_w_{k+1}'] + (1 - self.beta1) * self.gradients[f'w_{k+1}']
                  moments[f'm_b_{k+1}'] = self.beta1 * moments[f'm_b_{k+1}'] + (1 - self.beta1) * self.gradients[f'b_{k+1}']

                  velocities[f'v_w_{k+1}'] = self.beta2 * velocities[f'v_w_{k+1}'] + (1 - self.beta2) * (self.gradients[f'w_{k+1}'] ** 2)
                  velocities[f'v_b_{k+1}'] = self.beta2 * velocities[f'v_b_{k+1}'] + (1 - self.beta2) * (self.gradients[f'b_{k+1}'] ** 2)

                  # Bias correction
                  m_w_hat = moments[f'm_w_{k+1}'] / (1 - self.beta1 ** t)
                  m_b_hat = moments[f'm_b_{k+1}'] / (1 - self.beta1 ** t)

                  v_w_hat = velocities[f'v_w_{k+1}'] / (1 - self.beta2 ** t)
                  v_b_hat = velocities[f'v_b_{k+1}'] / (1 - self.beta2 ** t)

                  # Update parameters with Adam
                  self.parameters[f'w_{k+1}'] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                  self.parameters[f'b_{k+1}'] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

                  # t=t+1

          pred_labels = self.feed_forwards(self.train_data)
          tra_acc = self.accuracy(pred_labels,self.train_labels)
          tra_loss = self.loss_act_func.loss_functions(pred_labels, self.train_labels)

          pred_labels = self.feed_forwards(self.val_data)
          val_acc = self.accuracy(pred_labels,self.val_labels)
          val_loss = self.loss_act_func.loss_functions(pred_labels, self.val_labels)
          if(self.printFlag=="print"):
            print(f"Epoch {i+1}, Training Loss: {tra_loss} , Training Accuracy: {tra_acc} , Validation Loss: {val_loss} Validation Accuracy: {val_acc}")
          else:
            wandb.log({"Epoch": i+1, "Training_Loss": tra_loss, "Training_Accuracy": tra_acc,"Validation_Loss": val_loss,"Validation_Accuracy": val_acc})

      if(self.optimizer=="nadam"):
        '''
        Epoch 10, Loss: 0.645183860082157 Acc: 78.36333333333333
        TEST LOSS: 0.6794 ACCURACY: 77.5300
        '''
        moments={}
        for i in range(len(self.layer_sizes) - 1):
            moments[f'm_w_{i+1}'] = np.zeros((self.layer_sizes[i], self.layer_sizes[i+1]))
            moments[f'm_b_{i+1}'] = np.zeros((1, self.layer_sizes[i+1]))

        velocities={}
        for i in range(len(self.layer_sizes) - 1):
            velocities[f'v_w_{i+1}'] = np.zeros((self.layer_sizes[i], self.layer_sizes[i+1]))
            velocities[f'v_b_{i+1}'] = np.zeros((1, self.layer_sizes[i+1]))

        # t=0
        for i in range(self.epochs):
          t=1+i
          for j in range(0,self.no_of_samples,self.batch_size):
            # t=t+1
            start = j
            end = start + self.batch_size
            Y_train_labels = self.train_labels[start:end]
            X_train_images = self.train_data[start:end]

            pred_labels=self.feed_forwards(X_train_images)
            self.back_prop(X_train_images, Y_train_labels,pred_labels)

            for k in range(len(self.layer_sizes) - 1):

                self.gradients[f'w_{k+1}'] = self.gradients[f'w_{k+1}'] / self.batch_size
                self.gradients[f'b_{k+1}']=self.gradients[f'b_{k+1}']/self.batch_size

                self.gradients[f'w_{k+1}'] = self.gradients[f'w_{k+1}'] + self.weight_decay * self.parameters[f'w_{k+1}']

                # Update moments and velocities with Adam
                moments[f'm_w_{k+1}'] = self.beta1 * moments[f'm_w_{k+1}'] + (1 - self.beta1) * self.gradients[f'w_{k+1}']
                moments[f'm_b_{k+1}'] = self.beta1 * moments[f'm_b_{k+1}'] + (1 - self.beta1) * self.gradients[f'b_{k+1}']

                velocities[f'v_w_{k+1}'] = self.beta2 * velocities[f'v_w_{k+1}'] + (1 - self.beta2) * (self.gradients[f'w_{k+1}'] ** 2)
                velocities[f'v_b_{k+1}'] = self.beta2 * velocities[f'v_b_{k+1}'] + (1 - self.beta2) * (self.gradients[f'b_{k+1}'] ** 2)

                # Bias correction
                m_w_hat = moments[f'm_w_{k+1}'] / (1 - self.beta1 ** t)
                m_b_hat = moments[f'm_b_{k+1}'] / (1 - self.beta1 ** t)

                v_w_hat = velocities[f'v_w_{k+1}'] / (1 - self.beta2 ** t)
                v_b_hat = velocities[f'v_b_{k+1}'] / (1 - self.beta2 ** t)

                # Update parameters with NAdam
                nadam_factor = (1 - self.beta1) / (1 - (self.beta1 ** t))
                weigth_update = nadam_factor * self.gradients[f'w_{k+1}'] + (self.beta1 * m_w_hat)
                bias_update = nadam_factor * self.gradients[f'b_{k+1}'] + (self.beta1 * m_b_hat)
                self.parameters[f'w_{k+1}'] -= ((self.learning_rate / (np.sqrt(v_w_hat) + self.epsilon)) * weigth_update )
                self.parameters[f'b_{k+1}'] -= ((self.learning_rate / (np.sqrt(v_b_hat) + self.epsilon)) * bias_update )

            # t=t+1

          pred_labels = self.feed_forwards(self.train_data)
          tra_acc = self.accuracy(pred_labels,self.train_labels)
          tra_loss = self.loss_act_func.loss_functions(pred_labels, self.train_labels)

          pred_labels = self.feed_forwards(self.val_data)
          val_acc = self.accuracy(pred_labels,self.val_labels)
          val_loss = self.loss_act_func.loss_functions(pred_labels, self.val_labels)
          if(self.printFlag=="print"):
            print(f"Epoch {i+1}, Training Loss: {tra_loss} , Training Accuracy: {tra_acc} , Validation Loss: {val_loss} Validation Accuracy: {val_acc}")
          else:
            wandb.log({"Epoch": i+1, "Training_Loss": tra_loss, "Training_Accuracy": tra_acc,"Validation_Loss": val_loss,"Validation_Accuracy": val_acc})

#calculates the accuracy
  def accuracy(self, pred, truth):
    return ((np.argmax(truth, axis=1) == np.argmax(pred, axis=1)).mean())*100

# calculated loss and accuracy for test data
  def test(self, x, y):
    pred_labels=self.feed_forwards(x)
    loss = self.loss_act_func.loss_functions(pred_labels, y)
    acc = self.accuracy(pred_labels, y)
    print(f'TEST LOSS: {loss:.4f} TEST ACCURACY: {acc:.4f}')


class LossActFunc():
    #this class has methods for loss and activation functions
    def __init__(self, lossFun = "cross_entropy",actFun="sigmoid"):
        self.loss_func = lossFun
        self.act_fun = actFun

    #contains both the loss functions
    def loss_functions(self,pred, actual):
      if (self.loss_func=="cross_entropy"):
        return -np.sum(actual * np.log(pred + 1e-9)) / len(pred)

      elif (self.loss_func == "mean_squared_error"):
        squErr=(1/2)* np.sum((pred - actual) ** 2)
        meanSquErr=squErr/ len(pred)
        return  meanSquErr

    #this  methid returns the name of the loss function called
    def get_name(self):
      if (self.loss_func=="cross_entropy"):
        return "cross_entropy"

      elif (self.loss_func == "mean_squared_error"):
        return "mean_squared_error"

    #contains the activation functions and their derivatives
    def activation_func(self, x,real_der):
      #real_der=real --> want activation function
      #real_der=der --> want derivation of activation function
      if(self.act_fun == "sigmoid"):
        # sigmoid function
        if(real_der=="real"):
            return 1.0 / (1.0 + np.exp(-x))
        else:
           sigmoid_x = self.activation_func(x,"real")
           return sigmoid_x * (1 - sigmoid_x)
           

      if(self.act_fun == "tanh"):
        # tanh function
        if(real_der=="real"):
            return np.tanh(x)
        else:
           return 1 - (np.tanh(x) ** 2)
           

      if(self.act_fun == "ReLU"):
        if(real_der=="real"):
            return np.maximum(0, x)
        else:
            return (x > 0) * 1           


wandb.login(key='494428cc53b5c21da594f4fc75035d136c63a93c')
# take arguments passed while running this file
arguments = argparse.ArgumentParser()
arguments.add_argument('-wp' , '--wandb_project',type=str,default="CS6910 - Assignment 1")
arguments.add_argument('-we', '--wandb_entity' , type=str, default="Juhi_Zanje")
arguments.add_argument('-d', '--dataset', type=str, help="fashion_mnist, mnist",default="fashion_mnist")
arguments.add_argument('-e', '--epochs',  type=int, default=10)
arguments.add_argument('-b', '--batch_size', type=int, default=32)
arguments.add_argument('-l','--loss', type=str,help="cross_entropy,mean_squared_error", default="cross_entropy")
arguments.add_argument('-o', '--optimizer',type=str,help="sgd, momentum, nag, rmsprop, adam, nadam", default = "nadam")
arguments.add_argument('-lr', '--learning_rate',type=float, default=0.001)
arguments.add_argument('-m', '--momentum',type=float, default=0.9)
arguments.add_argument('-beta', '--beta', type=float, default=0.5)
arguments.add_argument('-beta1', '--beta1',type=float, default=0.09)
arguments.add_argument('-beta2', '--beta2',type=float, default=0.999)
arguments.add_argument('-eps', '--epsilon', type=float, default=1e-8)
arguments.add_argument('-w_d', '--weight_decay',type=float, default=0.0005)
arguments.add_argument('-w_i', '--weight_init',  type=str, help="random , Xavier",default="Xavier")
arguments.add_argument('-nhl', '--num_layers', type=int, default=4)
arguments.add_argument('-sz', '--hidden_size', type=int, default=256)
arguments.add_argument('-a', '--activation', help=" identity, sigmoid, tanh, ReLU", type=str, default="tanh")
ter_args = arguments.parse_args()

wandb.init(project= ter_args.wandb_project, name =ter_args.wandb_entity)

#Load dataset
if(ter_args.dataset=="fashion_mnist"):
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
else:
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], -1) / 255.0
test_images = test_images.reshape(test_images.shape[0], -1) / 255.0

#change trainlabel=2 to oneHot=[0 0 1 0 0 0 0 0 0 0]
train_labels_one_hot = np.eye(10)[train_labels]
test_labels_one_hot = np.eye(10)[test_labels]
# split train-data into train and validation data

train_images, val_images, train_labels_one_hot, val_labels_one_hot = train_test_split(train_images, train_labels_one_hot, test_size=0.1, random_state=42)

# Now we will set params and will train and test data
no_hidden_layers=ter_args.num_layers
hidden_layer_size=ter_args.hidden_size

layer_sizes=[]
layer_sizes.append(train_images.shape[1])
for i in range(no_hidden_layers):
    layer_sizes.append(hidden_layer_size)
layer_sizes.append(10)

print("layer_sizes=",layer_sizes)

loss_act_func = LossActFunc(ter_args.loss,ter_args.activation) #"mean_squared_error", "cross_entropy" # sigmoid, ReLU, tanh
# loss_func = LossFunc("cross_entropy")
epochs=ter_args.epochs
initialization=ter_args.weight_init #random , Xavier
learning_rate=ter_args.learning_rate
momentum=ter_args.momentum  #for momentum and nag
beta=ter_args.beta    #for rmsprop
beta1=ter_args.beta1    #for adam and nadam
beta2=ter_args.beta2    #for adam and nadam
epsilon=ter_args.epsilon
weight_decay=ter_args.weight_decay
batch_size=ter_args.batch_size
optimizer=ter_args.optimizer # sgd , momentum ,nag , rmsprop , adam , nadam
printFlag="wandb" #as we want to log to wandb

#ctreate object of MultiLayerPerceptron class 
mlp = MultiLayerPerceptron(train_images, train_labels_one_hot,val_images,val_labels_one_hot, layer_sizes,initialization,loss_act_func,batch_size,epochs,learning_rate,weight_decay,optimizer,momentum,beta,beta1,beta2,epsilon,printFlag)
#init wieghts with giveninitialization
mlp.init_weights()
#call given optimizer function to train 
mlp.optimizer_func()
#print Test loss and accuracy
mlp.test(test_images, test_labels_one_hot)

wandb.finish()





