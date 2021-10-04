import numpy as np

class Perceptron:
  def __init__(self, eta, epochs):     #eta = learning rate,   epochs = no of Passes
    self.weights = np.random.randn(3)*1e-4       #Random weight initialization
    print("Initial weight before Training:-\n", self.weights)
    self.eta = eta          
    self.epochs = epochs

  def activationfunction(self, inputs, weights):
    z = np.dot(inputs, weights)        # z = w * x in matrix format
    return np.where(z>0, 1, 0)         # 0 is threshold value here

  def fit(self, X, y):
    self.X=X
    self.y=y

    #bias = x * -1
    X_with_bias = np.c_[self.X, -np.ones((len(self.X),1))]      #concatination
    print("X with bias:-\n", X_with_bias)

    for epoch in range(self.epochs):
      print("--"*10)          #Separator for good visibility
      print("for epoch", epoch)
      print("--"*10)          #Separator for good visibility

      y_hat = self.activationfunction(X_with_bias, self.weights)  #forward propogation
      print("\nPredicted value after forward pass:-\n", y_hat)
      self.error = self.y - y_hat
      print("Error:-\n", self.error)
      self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error)    #backward propogation
      print(f"\nupdated weight after epoch:-: {epoch}/{self.epochs} : \n{self.weights}")
      print("#####"*10)

  def predict(self, X):
    X_with_bias = np.c_[X, -np.ones((len(X),1))]
    return self.activationfunction(X_with_bias, self.weights) 

  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"\nTotal loss: {total_loss}")
    return total_loss