using CSV, Plots

cd("Machine Learning")

data = CSV.File("wolfspider.csv")

X = data.feature

y_temp = data.class


Y = ifelse.(y_temp .== "present", 1, 0)

gr(size = (600, 600))

p_data = scatter(X, Y, 
xlabel = "Grain size (mm)", 
ylabel = "Wolf spider presence", 
legend = false, 
color = :red)

# initialize parameters
theta_0 = 0.0

theta_1 = 1.0

z(x) = theta_0 .+ theta_1 * x

# Hypothesis function
h(x) = 1 ./ (1 .+ exp.(-z(x)))

# Cost function
function cost()
    (-1 / m) * sum(
        Y .* log.(y_hat) + (1 .- Y) .* log.(1 .- y_hat)
        )
end

J_history = []

 # Partial derivative formulae from Andrew Ng
 function pd_theta_0()
    sum(y_hat - Y)
end

function pd_theta_1()
    sum((y_hat - Y) .* X)
end

epochs = 0

m = length(Y)
y_hat = h(X)

# Learning rates
alpha = 0.01


    y_hat = h(X)
    J = cost()

    plot!(X, y_hat, color = :green, linewidth = 1)
    y_hat = h(X)
    cost()

# Iterations
for i in 1:1000
    y_hat = h(X)
    J = cost()
    push!(J_history, J)

    plot!(0:0.1:1.2, h, color = :blue, linewidth = 1, alpha = 0.025)

    # Partial derivates for theta_0 and theta_1 for current iteration
    theta_0_temp = pd_theta_0()
    theta_1_temp = pd_theta_1()

    # Update parameters
    theta_0 -= alpha * theta_0_temp
    theta_1 -= alpha * theta_1_temp
end

p_data

h(1.0)
h(0.6)
h(0.2)