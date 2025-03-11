# Using Flux.jl to train a NN how to approximate the sin function
# FLOW Lab Onboarding
# Author: McKay Hoffmann
# Date: 3/7/2025

using Flux, Plots, ProgressMeter

####################################
# TRAINING DATA
####################################

x_train, x_test = Float32.(0:.1:20), 20:.1:40

y_train, y_test = sin.(x_train), sin.(x_test)

x_train_flux = reshape(x_train, 1, :)
####################################
# NEURAL NETWORK
####################################

# Dense(#Number of inputs => #Number of outputs)
model = Chain(
    Dense(1, 20),
    Dense(20, 20),
    Dense(20, 1)
)

model(x_train_flux)

####################################
# INITIALIZE PLOTS
####################################

Loss_History = zeros(30000)     # Keep track of loss to plot after training

titles = "Using Flux.jl. 2 layers, 20 neurons."

gr(size = (1000, 600))

# Plot true sin(x) function
progress = plot(x_train, y_train,     
title = titles,
legend = :topright,  
label = "True function",
color = :red,
lw = 2
)

# Call neural network.
y_hat = model(x_train_flux)
y_hat_graphable = vec(y_hat)

# Plot initial model for comparison
plot!(x_train, y_hat_graphable,
label = "Initial model",
color = :blue,
alpha = 0.5)

####################################
# PREPARE LOSS FUNCTIONS
####################################

# define loss function
y_train = reshape(y_train, 1, :)
loss(model, x, y) = Flux.Losses.mse(model(x), y)

# Format data
data = [(x_train_flux, y_train)]

# select optimizer
learning_rate = 0.1

opt = Flux.setup(ADAM(learning_rate), model)

loss(model, x_train_flux, y_train)

epochs = 5000

for epoch in 1:epochs
    # train model
    Flux.train!(loss, model, data, opt)
    # print report
    train_loss = loss(model, x_train_flux, y_train)
    Loss_History[epoch + 5000] = train_loss
    println("Epoch = $(epoch + 5000)  : Training Loss = $train_loss")
end

####################################
# PLOT PROGRESS
####################################

# Call neural network.
y_hat = model(x_train_flux)
y_hat_graphable = vec(y_hat)

# Plot initial model for comparison
plot!(x_train, y_hat_graphable,
label = "10,000 epochs",
color = :green,
alpha = 0.5)

plot(1:10000, Loss_History[1:10000],
title = titles,
label = "Loss History",
color = :blue,
legend = :topright
)