################################################################################
# ML > Supervised Learning > Artificial Neural Network
################################################################################

# load packages

using Flux, Images, MLDatasets, Plots

using Flux: crossentropy, onecold, onehotbatch, params, train!

using Random, Statistics

# set random seed

Random.seed!(1)

# load data

X_train_raw, y_train_raw = MLDatasets.MNIST(:train)[:]

X_test_raw, y_test_raw = MLDatasets.MNIST(:test)[:]

# view training input

X_train_raw

index = 5

img = X_train_raw[:, :, index]

colorview(Gray, img')

# view training label

y_train_raw

y_train_raw[index]

# view testing input

X_test_raw

img = X_test_raw[:, :, index]

colorview(Gray, img')

# view testing label

y_test_raw

y_test_raw[index]

# flatten input data

X_train = Flux.flatten(X_train_raw)

X_test = Flux.flatten(X_test_raw)

# one-hot encode labels

y_train = onehotbatch(y_train_raw, 0:9)

y_test = onehotbatch(y_test_raw, 0:9)

# define model architecture

model = Chain(
    Dense(28 * 28, 32, relu),
    Dense(32, 10),
    softmax
)

# define loss function

loss(model, x, y) = Flux.Losses.crossentropy(model(x), y)

# Format data

data = [(X_train, y_train)]

# select optimizer

learning_rate = 0.01

opt = Flux.setup(ADAM(learning_rate), model)

# train model

loss_history = []

epochs = 500

loss(model, X_train, y_train)

for epoch in 1:epochs
    # train model
    train!(loss, model, data, opt)
    # print report
    train_loss = loss(model, X_train, y_train)
    push!(loss_history, train_loss)
    println("Epoch = $epoch : Training Loss = $train_loss")
end

# make predictions

y_hat_raw = model(X_test)

y_hat = onecold(y_hat_raw) .- 1

y = y_test_raw

mean(y_hat .== y)

mean(y_hat .== y)

mean(y_hat .== y)


# display results

check = [y_hat[i] == y[i] for i in 1:length(y)]

index = collect(1:length(y))

check_display = [index y_hat y check]

vscodedisplay(check_display)

# view misclassifications

misclass_index = 150

img = X_test_raw[:, :, misclass_index]

colorview(Gray, img')

y[misclass_index]

y_hat[misclass_index]

# initialize plot

gr(size = (600, 600))

# plot learning curve

p_l_curve = plot(1:700, loss_history,
    xlabel = "Epochs",
    ylabel = "Loss",
    title = "Learning Curve",
    legend = false,
    color = :blue,
    linewidth = 2
)

p_l_curve
# save plot

savefig(p_l_curve, "ann_learning_curve.svg")