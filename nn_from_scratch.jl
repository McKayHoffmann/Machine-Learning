# Simple neural network from scratch
# FLOW Lab Onboarding
# Author: McKay Hoffmann
# Date: 2/24/2025

using ForwardDiff, Plots

# Intialize training data
x_train, x_test = 0:.1:10, 10:.1:20

y_train, y_test = sin.(x_train), sin.(x_test)

gr(size = (600, 600))
progess = plot(x_train, y_train,
label = "True function")

# Initialze parameters
phi = zeros(16)

function activation(z)
    return 1 / (1 + exp(-z))
end

function nn(input_data, phi)
    w_0 = phi[1:5]
    w_1 = phi[6:10]
    b_1 = phi[11:15]
    b_2 = phi[16]
    z1 = w_0 * input_data + b_1
    a1 = activation.(z1)
    y_hat = w_1' * a1 + b_2
    return y_hat
end

# Call neural network.
y_hat = nn.(x_train, Ref(phi))      # Returns vector y_hat

# Plot current model for comparison
plot!(x_train, y_hat,
label = "Neural Network",
color = :red,
alpha = 0.1)

# Define loss function
function Loss(x_train, y_train, phi)
    y_hat = nn.(x_train, Ref(phi))
    return sum((y_hat .- y_train) .^ 2) / length(y_train)     # Mean squared error
end

# Enclose loss function for use in ForwardDiff
function wrapper(x_train, y_train)
    function wrapped_loss(phi)
        return Loss(x_train, y_train, phi)
    end
    return wrapped_loss     # Return new 
end

wrapped_loss = wrapper(x_train, y_train)

# Calculate gradient using ForwardDiff
grad_Loss = ForwardDiff.gradient(wrapped_loss, phi)

# Change phi according to the gradient
LR = 0.1     # Learning rate

# Loss history
Loss_History = zeros(5000)

for epochs in 1:5000
    grad_Loss = ForwardDiff.gradient(wrapped_loss, phi)
    phi = phi - (LR * grad_Loss)
    Loss_History[epochs] = Loss(x_train, y_train, phi)

    # Call neural network.
    y_hat = nn.(x_train, Ref(phi))      # Returns vector y_hat

    # Plot current model for comparison
    plot!(x_train, y_hat,
    label = false,
    color = :red,
    alpha = 0.1
    )
end

plot!(x_train, y_hat,
    title = "Epochs: 5000. 1 layer, 5 neurons.",
    label = false,
    color = :red,
    alpha = 0.1
)

savefig(progess, "1 layer, 5 neurons")