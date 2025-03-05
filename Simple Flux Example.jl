using Flux

actual(x) = x^2 + 5x - 10

x_train, x_test = hcat(0:5)', hcat(6:10)'

y_train, y_test = actual.(x_train), actual.(x_test)

# Dense(#Number of inputs => #Number of outputs)
model = Chain(
    Dense(1, 2),
    Dense(2, 1)
)

model(x_train)

y_train

loss(model, x, y) = Flux.Losses.mse(model(x), y)

loss(model, x_train, y_train)
using Flux: train!

# Declare optimizing algorithm
opt = Descent()

# Prepare data in the form of tuples
data = [(x_train, y_train)]

train!(loss, model, data, opt)

loss(model, x_train, y_train)

for epoch in 1:200
    train!(loss, model, data, opt)
end

loss(model, x_train, y_train)

round.(Int, model(x_test))
y_test

model.weight
model.bias

################################################
# Own example from scratch
################################################

using Flux

actual_f(x) = x^2 + 5x - 10

x_train, x_test = hcat(0:5)', hcat(6:10)'

y_train, y_test = actual_f.(x_train), actual_f.(x_test)

new_model = Dense(1, 1)

new_loss(::Any, ::Any) = Flux.Losses.mse(model(x), y)

loss(x_train, y_train)

opt = Descent()

new_data = [(x_train, y_train)]

train!(new_loss, new_model, new_data, opt)