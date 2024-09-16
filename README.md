# California Housing Price Prediction with PyTorch 🏠📈

This notebook demonstrates a simple neural network built with PyTorch to predict California housing prices.  We'll walk through the process step-by-step, making it easy to understand even if you're new to machine learning! 

**🚀 What we'll do:**

1. **Gather our data:** We'll use the famous California Housing dataset, which contains information about houses in California.
2. **Prepare the data:**  We'll split the data into training and testing sets and scale the features to help our model learn better.
3. **Build our model:** We'll create a neural network with a few layers to learn the complex relationships in the data.
4. **Train our model:** We'll feed the training data to our model and adjust its parameters to make accurate predictions.
5. **Evaluate our model:** We'll see how well our model performs on unseen data (the test set).
6. **Save our model:** We'll save our trained model so we can use it later!


**💡 Key Concepts:**

* **PyTorch:** A powerful library for building and training neural networks. It provides tools for defining models, optimizing them, and working with data.
* **Neural Networks:**  A type of machine learning model inspired by the human brain. They learn patterns in data by adjusting the connections between artificial neurons.
* **Regression:**  A type of machine learning where we predict a continuous value (like house prices).
* **Mean Squared Error (MSE):** A common way to measure how well a regression model performs. It calculates the average squared difference between the predicted values and the actual values.


**💻 Code Breakdown:**

**1. Loading the Data:**

   * We use the `fetch_california_housing` function from `sklearn.datasets` to load the dataset.
   * The data is then converted to Pandas DataFrames for easier exploration.
   * We display a sneak peek of the data to see what we're working with!

**2. Preparing the Data:**

   * **Splitting:** We divide the data into training and testing sets. This helps us evaluate how well our model generalizes to new data.
   * **Scaling:** We standardize the input features using `StandardScaler` from `sklearn.preprocessing`. This ensures that all features have a similar range of values, which can improve the training process.
   * **PyTorch Tensors:** We convert the data into PyTorch tensors, which are the fundamental building blocks for working with data in PyTorch.

**3. Building the Model:**

   * **`RegressionModel` class:** We define a simple neural network with multiple layers.
     * `nn.Linear`:  These are the layers that connect the neurons.
     * `nn.ReLU`: This is an activation function that introduces non-linearity into our model.
   * **Creating the model:** We create an instance of our `RegressionModel` class and specify the input size.

**4. Training the Model:**

   * **Loss function:** We use `nn.MSELoss` to calculate the difference between our model's predictions and the actual values.
   * **Optimizer:** We use `optim.SGD` (Stochastic Gradient Descent) to adjust the model's parameters during training.
   * **Training loop:** We iterate through the training data multiple times (epochs). In each epoch:
     * **Forward pass:** We pass the input data through the model to get predictions.
     * **Calculate loss:** We calculate the loss between the predictions and the actual values.
     * **Backward pass:** We calculate the gradients (how much to adjust the parameters).
     * **Optimization:** We update the model's parameters using the gradients.

**5. Evaluating the Model:**

   * **Saving the model:** We save the trained model to a file named `california_housing_model.pth`.
   * **Loading the model:** We load the saved model for future use.
   * **Testing:** We evaluate the loaded model on the test set and calculate the mean squared error (MSE).


**✨ Further Improvements:**

* **Experiment with different architectures:** Try adding more layers, different activation functions, or different types of layers.
* **Hyperparameter tuning:** Adjust the learning rate, number of epochs, and other parameters to find the best model.
* **Data visualization:** Create plots to understand the data better and visualize the model's predictions.
* **Regularization:** Add techniques like dropout or L2 regularization to prevent overfitting.
* **More complex models:** Explore more advanced architectures like convolutional neural networks (CNNs) or recurrent neural networks (RNNs) if needed.


**🤝 Let's learn together!**

If you have any questions, feel free to reach out! Happy coding! 💻
