import math
import torch
import gpytorch
import matplotlib.pyplot as plt
import os

# Setting up training data

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0,1,100)

# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2*math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        
        # You're customizing a car (your ExactGPModel), but you still need to start the engine and set up the basics, that’s what the super().__init__() line does.
        # super().__init__() in ExactGPModel calls the parent ExactGP constructor to hook in the training data, likelihood, and all the math needed to do GP inference.
        super().__init__(train_x, train_y, likelihood) 
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 50

# Find optimial model hyperparameters

# Setting the model in training mode.
#Important because some components (e.g., dropout, batch normalization, or noise estimation) behave differently during training versus evaluation.
model.train()
likelihood.train()

# Use the atom optimizer
# Creates an optimizer that knows how to update every tunable part of your model, including:
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1) # Includes GaussianLikelihood parameters

# Loss for GPs: the marginal log likelihood
# This sets up the Exact Marginal Log Likelihood (MLL) as the loss function for training.
# Likelihood is the Gaussian likelihood (e.g., with a learnable noise term).
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Training
for i in range(training_iter):
    
    # Zero gradients from previous iteration (Clears out old gradients from the last iteration)
    optimizer.zero_grad()
    
    # Output from model
    # model(train_x) → model.__call__(train_x) → model.forward(train_x)

    output = model(train_x)
    
    # Calc loss and backprop gradients
    # Why these innputs? Because the marginal log likelihood compares: (1) what your model predicts (output) against (2) what you actually observed (train_y)
    # And gives you a scalar value (the log probability) representing how well the model explains the data (higher is better)
    # loss is a scalar tensor
    loss = -mll(output, train_y)
    
    # When you call .backward() on a scalar: It starts from that scalar and uses the chain rule to compute gradients all the way back to the model parameters.
    # loss is our final scalar result
    loss.backward()
    
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    
    # Updates the model's parameters using the gradients you've computed with loss.backward().
    optimizer.step()
    
# Get into evaluation (predictive pos terior) mode
model.eval()
likelihood.eval()

# Make predictions by feeding model through likelihood

# torch.no_grad() tells pytorch not to compute gradients. this makes prediction faster and uses less memory
#During training, gradients are needed (for .backward() and optimizer.step()). During evaluation/prediction, you're just using the model, no need for gradients.

# .fast_pred_var() is a gpytorch-specific setting that speeds up prediction
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    
    # Test points are regularly spaced along [0,1]
    test_x = torch.linspace(0,1,51)
    
    # Model's predictive distribution over y, the observed outputs. Therefore, observed_pred is a distribution
    # model(test_x) gives posterior distribution over the latent function f(x). Then, likelihood transforms f(x) into predicted observed outputs y(x) = f(x) + ε
    observed_pred = likelihood(model(test_x))
    
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])