# Simple neural network from scratch
# FLOW Lab Onboarding
# Author: McKay Hoffmann
# Date: 2/24/2025

############ Things to Try ############
# Tanh instead of sigmoid -> Didn't work
# Try a smaller learning rate and graph learning rate over epochs (graph learning rate over epochs first) -> Didn't work
# What about a larger learning rate? -> Didn't work
# Try a different optimizer (ADAM instead of gradient descent)

using ForwardDiff, Plots
using ProgressMeter

# Intialize training data
x_train, x_test = 0:.1:20, 20:.1:40

y_train, y_test = sin.(x_train), sin.(x_test)

gr(size = (1000, 600))
progress = plot(x_train, y_train,
label = "True function",
color = :red,
lw = 2
)

# Initialze parameters
phi = rand(141)


function activation(z)      # RELU
    # return (exp(z) - exp(-z)) / (exp(z) + exp(-z))
    return 1 / (1 + exp(-z))
    # if z < 0
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

# Call neural network.
y_hat = nn.(x_train, Ref(phi))      # Returns vector y_hat

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
# Initialize Momentum
####################################
vk_1 = 0        # vk-1 is 0 for the first epoch
beta = 0.4      # Hypervariable for momentum
LR = 0.1     # Learning rate

# Loss history
Loss_History = zeros(30000)

##########################################################
# BEGIN TRAINING
##########################################################


#############################
# 1st 5000
#############################
@showprogress for epochs in 1:5000
    grad = ForwardDiff.gradient(wrapped_loss, phi)      # Calculate the gradient of loss with respect to phi
    vk = ((beta * vk_1) .+ ((1 - beta)grad)) / (1 - (beta)^epochs)       # Momentum
    vk_1 = vk
    phi = phi - (LR * vk)       # Take a step

    # Record Loss
    Loss_History[epochs] = Loss(x_train, y_train, phi)
end

# Loss_History

plot!(x_train, y_hat,
    label = "5000 epochs",
    color = :orange,
    lw = 1,
    alpha = 0.5
)

#############################
# 2nd 5000
#############################
@showprogress for epochs in 1:5000
    grad_Loss = ForwardDiff.gradient(wrapped_loss, phi)
    phi = phi - (LR * grad_Loss)

    # Record Loss
    Loss_History[epochs + 5000] = Loss(x_train, y_train, phi)

    # Call neural network.
    y_hat = nn.(x_train, Ref(phi))      # Returns vector y_hat

end

plot!(x_train, y_hat,
    label = "10000 epochs",
    color = :green,
    lw = 1,
    alpha = 1
)

#############################
# 3rd 5000
#############################
@showprogress for epochs in 1:5000
    grad_Loss = ForwardDiff.gradient(wrapped_loss, phi)
    phi = phi - (LR * grad_Loss)

    # Record Loss
    Loss_History[epochs + 10000] = Loss(x_train, y_train, phi)

    # Call neural network.
    y_hat = nn.(x_train, Ref(phi))      # Returns vector y_hat

end

plot!(x_train, y_hat,
    label = "15000 epochs",
    color = :purple,
    lw = 1,
    alpha = 0.5
)

#############################
# 4th 5000
#############################
@showprogress for epochs in 1:5000
    grad_Loss = ForwardDiff.gradient(wrapped_loss, phi)
    phi = phi - (LR * grad_Loss)

    # Record Loss
    Loss_History[epochs + 15000] = Loss(x_train, y_train, phi)

    # Call neural network.
    y_hat = nn.(x_train, Ref(phi))      # Returns vector y_hat

end

plot!(x_train, y_hat,
    label = "20000 epochs",
    color = :black,
    lw = 1,
    alpha = 0.5
)

#############################
# 5th 5000
#############################
@showprogress for epochs in 1:5000
    grad_Loss = ForwardDiff.gradient(wrapped_loss, phi)
    phi = phi - (LR * grad_Loss)

    # Record Loss
    Loss_History[epochs + 20000] = Loss(x_train, y_train, phi)

    # Call neural network.
    y_hat = nn.(x_train, Ref(phi))      # Returns vector y_hat

end

plot!(x_train, y_hat,
    label = "25000 epochs",
    color = :cyan,
    lw = 2,
    alpha = 0.5
)

#############################
# 6th 5000
#############################
@showprogress for epochs in 1:5000
    grad_Loss = ForwardDiff.gradient(wrapped_loss, phi)
    phi = phi - (LR * grad_Loss)

    # Record Loss
    Loss_History[epochs + 20000] = Loss(x_train, y_train, phi)

    # Call neural network.
    y_hat = nn.(x_train, Ref(phi))      # Returns vector y_hat

end

plot!(progress, x_train, y_hat,
    title = "LR = 0.1, Beta = 0.4",
    label = "30000 epochs",
    color = :magenta,
    lw = 2,
    alpha = 1,
    legend = :topright
)

############# Extra 20,000 epochs #################
# @showprogress for epochs in 1:20000
#     grad_Loss = ForwardDiff.gradient(wrapped_loss, phi)
#     phi = phi - (LR * grad_Loss)

#     # Record Loss
#     Loss_History[epochs + 25000] = Loss(x_train, y_train, phi)

#     # Call neural network.
#     y_hat = nn.(x_train, Ref(phi))      # Returns vector y_hat
# end

# plot!(x_train, y_hat,
#     label = "50000 epochs",
#     color = :black,
#     lw = 1,
#     alpha = 0.5
# )

Loss_plot = plot(1:30000, Loss_History[1:30000],
title = "Loss History: LR = 0.1, Beta = 0.4",
label = "Loss",
color = :blue
)

savefig(Loss_plot, "Loss History LR 0,1 Beta 0,4")
savefig(progress, "Model LR 0,1 Beta 0,4")