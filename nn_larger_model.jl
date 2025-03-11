# Simple neural network from scratch
# FLOW Lab Onboarding
# Author: McKay Hoffmann
# Date: 2/24/2025

############ Things to Try ############
# Train it giving more data every so many epochs. (Start with 1 period and give it more periods during training.)
# Make several different models for each portions

########### To do after #############
# Teach a machine learning model to predict Hess-Smith panel method w/ Cl, Cd, Cm.
    # Collect data and put it into a file. (DelimitedFiles)
# A big plus to training a neural net is that you can find an gradient-based optimum.

using Plots, ProgressMeter, Random, Distributions, ReverseDiff

####################################
# TRAINING DATA
####################################
x_train_1, x_train_2, x_train_3, x_train_4 = 0:.1:5, 5.1:0.1:10, 10.1:0.1:15, 15.1:0.1:20

y_train_1, y_train_2, y_train_3, y_train_4 = sin.(x_train_1), sin.(x_train_2), sin.(x_train_3), sin.(x_train_4)

####################################
# NEURAL NETWORK
####################################

### Initialize parameters ###
n_in = 50       # Neurons in
n_out = 50      # Neurons out
phi = zeros(10351)      # 1 input, 1 output, 5 layers, 50 neurons each

### Glorot initialization ###

phi[1:10100] .= rand.(Normal(0, sqrt(2 / (n_in + n_out))))      # Initialize weights, leave biases as zeros

### Activation function ###
function activation(z)      # Current activation: SIGMOID
    # return (exp(z) - exp(-z)) / (exp(z) + exp(-z))        # Tanh
    return 1 / (1 + exp(-z))                                # Sigmoid
    # if z < 0                                              # RELU
    #     return 0
    # else
    #     return z
    # end
end

### Neural Network ###
function nn(input_data, phi)        # 5 layers, 50 neurons each
    w_0 = phi[1:50]     # Notation: w_N -> N = Layer of origin. Example: w_o -> Weights from layer 0 (inputs) to layer 1.
    w_1 = reshape(phi[51:2550], 50, 50)      # 50 x 50 matrix
    w_2 = reshape(phi[2551:5050], 50, 50)
    w_3 = reshape(phi[5051:7550], 50, 50)
    w_4 = reshape(phi[7551:10050], 50, 50)
    w_5 = phi[10051:10100]      # 50 element vector
    b_1 = phi[10101:10150]
    b_2 = phi[10151:10200]
    b_3 = phi[10201:10250]
    b_4 = phi[10251:10300]
    b_5 = phi[10301:10350]
    b_6 = phi[10351]
    z1 = w_0 * input_data + b_1 # Input data = x
    a1 = activation.(z1)        # Activation of layer 1
    z2 = w_1 * a1 + b_2
    a2 = activation.(z2)        # Activation of layer 2
    z3 = w_2 * a2 + b_3
    a3 = activation.(z3)        # Activation of layer 3
    z4 = w_3 * a3 + b_4
    a4 = activation.(z4)        # Activation of layer 4
    z5 = w_4 * a4 + b_5
    a5 = activation.(z5)        # Activation of layer 5
    y_hat = w_5' * a5 + b_6     # Final output (layer 6)
    return y_hat
end

####################################
# INITIALIZE PLOTS
####################################

Loss_History = zeros(30000)     # Keep track of loss to plot after training

titles = "5 layers, 50 neurons each.\nUsing ReverseDiff. Momentum: LR = 0.01, B = 0.1"

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
y_hat = nn.(x_train, Ref(phi))

# Plot initial model for comparison
plot!(x_train, y_hat,
label = "Initial model",
color = :blue,
alpha = 0.5)

####################################
# PREPARE LOSS FUNCTIONS
####################################

# Mean squared error (MSE) loss function
function Loss(x_train, y_train, phi)
    y_hat = nn.(x_train, Ref(phi))
    return sum((y_hat .- y_train) .^ 2) / length(y_train)
end

# Enclose loss function for use in ReverseDiff (wrapped_loss can only be a function of phi)
function wrapper(x_train, y_train)
    function wrapped_loss(phi)      # Captures x_train and y_train inside of its scope, only takes phi as an argument
        return Loss(x_train, y_train, phi)      # Return what calling Loss would have
    end
    return wrapped_loss
end

####################################
# OPTIMIZATION STRATEGY: MOMENTUM
####################################
vk_1 = 0        # vk-1 is 0 only for the first epoch, cannot include in a for loop
beta = 0.1      # Hypervariable for momentum. (What percent of the old gradient is kept)
LR = 0.01        # Learning rate

###################################
# ReverseDiff.jl
###################################

wrapped_loss = wrapper(x_train, y_train) # Input

# pre-record a GradientTape for wrapped_loss
const loss_tape = ReverseDiff.GradientTape(wrapped_loss, phi)

# compile 'loss_tape' into a more optimized representation
const compiled_loss_tape = ReverseDiff.compile(loss_tape)

######################################################################
# BEGIN TRAINING
######################################################################

# Training function to be looped. Returns vk_1 and phi for future use.
function train!(phi, epoch, beta, vk_1, LR, compiled_loss_tape, wrapped_loss)
    grad = ReverseDiff.gradient!(compiled_loss_tape, phi)
    vk = ((beta * vk_1) .+ ((1 - beta)grad)) / (1 - (beta)^epoch)       # Momentum
    vk_1 = vk
    phi = phi - (LR * vk)

    # Record Loss
    Loss_History[epoch] = wrapped_loss(phi)

    return vk_1, phi
end

#   1st 5000   #
################
@showprogress for epoch in 1:5000
    vk_1, phi = train!(phi, epoch, beta, vk_1, LR, compiled_loss_tape, wrapped_loss)
end

y_hat = nn.(x_train, Ref(phi))

plot!(x_train, y_hat,
    label = "5000 epochs",
    color = :orange,
    lw = 1,
    alpha = 0.5
)

### 2nd 5000 ###
################
@showprogress for epoch in 5001:10000
    vk_1, phi = train!(phi, epoch, beta, vk_1, LR, compiled_loss_tape, wrapped_loss)
end

y_hat = nn.(x_train, Ref(phi))

plot!(x_train, y_hat,
    label = "10000 epochs",
    color = :green,
    lw = 1,
    alpha = 0.5
)

### 3rd 5000 ###
################
@showprogress for epoch in 10001:15000
    vk_1, phi = train!(phi, epoch, beta, vk_1, LR, compiled_loss_tape, wrapped_loss)
end

y_hat = nn.(x_train, Ref(phi))

plot!(x_train, y_hat,
    label = "15000 epochs",
    color = :cyan,
    lw = 1,
    alpha = 0.5
)

### 4th 5000 ###
################
@showprogress for epoch in 15001:20000
    vk_1, phi = train!(phi, epoch, beta, vk_1, LR, compiled_loss_tape, wrapped_loss)
end

y_hat = nn.(x_train, Ref(phi))

plot!(x_train, y_hat,
    label = "20000 epochs",
    color = :purple,
    lw = 1,
    alpha = 0.5
)

### 5th 5000 ###
################
@showprogress for epoch in 20001:25000
    vk_1, phi = train!(phi, epoch, beta, vk_1, LR, compiled_loss_tape, wrapped_loss)
end

y_hat = nn.(x_train, Ref(phi))

plot!(x_train, y_hat,
    label = "25000 epochs",
    color = :magenta,
    lw = 1,
    alpha = 0.5
)

### 6th 5000 ###
################
@showprogress for epoch in 25001:30000
    vk_1, phi = train!(phi, epoch, beta, vk_1, LR, compiled_loss_tape, wrapped_loss)
end

y_hat = nn.(x_train, Ref(phi))

plot!(progress, x_train, y_hat,
    label = "30000 epochs / Final model",
    color = :black,
    lw = 2,
    alpha = 1
)

######################################################################
# END TRAINING
######################################################################

# Create Loss plot for evaluating training
Loss_plot = plot(1:30000, Loss_History[1:30000],
title = titles,
label = "Loss",
xlabel = "Epochs",
color = :blue
)

# Save plots for comparing experiments
savefig(Loss_plot, "Loss History LR 0,01 Beta 0,1")
savefig(progress, "Model LR 0,01 Beta 0,1")