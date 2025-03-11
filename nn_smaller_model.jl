# Simple neural network from scratch
# FLOW Lab Onboarding
# Author: McKay Hoffmann
# Date: 2/24/2025

using Plots, ProgressMeter, Random, Distributions, ReverseDiff

####################################
# TRAINING DATA
####################################
x_train, x_test = 0:.1:2pi, 0:0.1:2pi

y_train, y_test = sin.(x_train), sin.(x_test)

####################################
# NEURAL NETWORK
####################################

### Initialize parameters ###
n_in = 10       # Neurons in
n_out = 10      # Neurons out
phi = zeros(141)      # 1 input, 1 output, 2 layers, 10 neurons each

### Glorot initialization ###

phi[1:120] .= rand.(Normal(0, sqrt(2 / (n_in + n_out))))      # Initialize weights, leave biases as zeros

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

function nn(input_data, phi)        # 2 layers, 10 neurons each
    w_0 = phi[1:10]
    w_1 = reshape(phi[11:110], 10, 10)      # 100 x 100
    w_2 = phi[111:120]
    b_1 = phi[121:130]
    b_2 = phi[131:140]
    b_3 = phi[141]
    z1 = w_0 * input_data + b_1
    a1 = activation.(z1)
    z2 = w_1 * a1 + b_2
    a2 = activation.(z2)
    y_hat = w_2' * a2 + b_3
    return y_hat
end

####################################
# INITIALIZE PLOTS
####################################

Loss_History = zeros(500_000)     # Keep track of loss to plot after training

titles = "2 layers, 10 neurons each.\nUsing ReverseDiff. Momentum: LR = 0.1, B = 0.2"

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

# Define loss function
function Loss(x_train, y_train, phi)
    y_hat = nn.(x_train, Ref(phi))
    return sum((y_hat .- y_train) .^ 2) / length(y_train)     # Mean squared error
end

# Enclose loss function for use in ForwardDiff
function wrapper(x_train, y_train)
    function wrapped_loss(phi)      # Captures x_train and y_train inside of its scope, only takes phi as an argument
        return Loss(x_train, y_train, phi)      # Give all three to the Loss function
    end
    return wrapped_loss
end

wrapped_loss = wrapper(x_train, y_train)        # Wrapped_loss is now a function of only phi

####################################
# OPTIMIZATION STRATEGY: MOMENTUM
####################################
vk_1 = 0        # vk-1 is 0 only for the first epoch, cannot include in a for loop
beta = 0.2      # Hypervariable for momentum. (What percent of the old gradient is kept)
LR = 0.1        # Learning rate

###################################
# ReverseDiff.jl
###################################

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

#   1st 20000   #
################
@showprogress for epoch in 1:100000
    vk_1, phi = train!(phi, epoch, beta, vk_1, LR, compiled_loss_tape, wrapped_loss)
end

y_hat = nn.(x_train, Ref(phi))

plot(x_train, y_hat,
    label = "100,000 epochs",
    color = :orange,
    lw = 1,
    alpha = 0.5
)

### 2nd 20000 ###
################
@showprogress for epoch in 100001:200000
    vk_1, phi = train!(phi, epoch, beta, vk_1, LR, compiled_loss_tape, wrapped_loss)
end

y_hat = nn.(x_train, Ref(phi))

plot!(x_train, y_hat,
    label = "200,000 epochs",
    color = :green,
    lw = 1,
    alpha = 0.5
)

Loss(x_train, y_train, phi)

### 3rd 20000 ###
################
@showprogress for epoch in 200001:300000
    vk_1, phi = train!(phi, epoch, beta, vk_1, LR, compiled_loss_tape, wrapped_loss)
end

y_hat = nn.(x_train, Ref(phi))

plot!(x_train, y_hat,
    label = "300,000 epochs",
    color = :cyan,
    lw = 1,
    alpha = 0.5
)

### 4th 20000 ###
################
@showprogress for epoch in 300001:400000
    vk_1, phi = train!(phi, epoch, beta, vk_1, LR, compiled_loss_tape, wrapped_loss)
end

y_hat = nn.(x_train, Ref(phi))

plot(x_train_3, y_hat,
    label = "400,000 epochs",
    color = :purple,
    lw = 1,
    alpha = 0.5
)

### 5th 20000 ###
################
@showprogress for epoch in 400001:500000
    vk_1, phi = train!(phi, epoch, beta, vk_1, LR, compiled_loss_tape, wrapped_loss)
end

y_hat = nn.(x_train, Ref(phi))

plot!(x_train, y_hat,
    label = "500,000 epochs / Final model",
    color = :black,
    lw = 2,
    alpha = 1
)

######################################################################
# END TRAINING
######################################################################

# Create Loss plot for evaluating training
Loss_plot = plot(1:300000, Loss_History[1:300000],
title = titles,
label = "Loss",
xlabel = "Epochs",
color = :blue
)

# Save plots for comparing experiments
savefig(Loss_plot, "Loss History LR 0,1 Beta 0,2 300,000 epochs")
savefig(progress, "Model LR 0,1 Beta 0,2 300,000 epochs")
