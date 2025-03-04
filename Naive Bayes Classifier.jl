# Multivariable classification using Naive Bayes Classifier
# Author: doggo dot jl
# Date: 7 Feb 2025

# Data Wrangling

using DelimitedFiles

data = readdlm("tennis.csv", ',', skipstart= 1) # skipstart is a command that skips the first row of the data

# split matrix into vectors (data is a matrix)

vscodedisplay(data)

x1 = data[:, 1] # Outlook
x2 = data[:, 2] # Temperature
x3 = data[:, 3] # Humidity
x4 = data[:, 4] # Wind

y = data[:, 5] # Play

# identify unique elements

uniq_x1 = unique(x1)
uniq_x2 = unique(x2)
uniq_x3 = unique(x3)
uniq_x4 = unique(x4)

uniq_y = unique(y)

# calculate probabilities for "yes" and "no" outputs

len_y = length(y)

len_yes = count(x -> x == "yes", y)

len_no = count(x -> x == "no", y)

p_yes = len_yes / len_y

p_no = len_no / len_y

# split "yes" and "no" into seperate matrices

data_yes = data[data[:,5] .== "yes", :]

data_no = data[data[:,5] .== "no", :]

# count features in data_yes

len_sunny_yes = count(x -> x == "sunny", data_yes)
len_overcast_yes = count(x -> x == "overcast", data_yes)
len_rainy_yes = count(x -> x == "rainy", data_yes)

len_hot_yes = count(x -> x == "hot", data_yes)
len_mild_yes = count(x -> x == "mild", data_yes)
len_cool_yes = count(x -> x == "cool", data_yes)

len_high_yes = count(x -> x == "high", data_yes)
len_normal_yes = count(x -> x == "normal", data_yes)

len_not_windy_yes = count(x -> x == false, data_yes)
len_windy_yes = count(x -> x == true, data_yes)

# count features in data_no

len_sunny_no = count(x -> x == "sunny", data_no)
len_overcast_no = count(x -> x == "overcast", data_no)
len_rainy_no = count(x -> x == "rainy", data_no)

len_hot_no = count(x -> x == "hot", data_no)
len_mild_no = count(x -> x == "mild", data_no)
len_cool_no = count(x -> x == "cool", data_no)

len_high_no = count(x -> x == "high", data_no)
len_normal_no = count(x -> x == "normal", data_no)

len_not_windy_no = count(x -> x == false, data_no)
len_windy_no = count(x -> x == true, data_no)

##################################################################
# Naive Bayes Classifier
##################################################################

#                 P(B | A) * P(A)
#   P(A | B) =  -------------------
#                      P(B)

# Prediction 1: newX = ["sunny", "hot"]

p_yes_newX =
    (len_sunny_yes / len_yes) *
    (len_hot_yes / len_yes) *
    p_yes

p_no_newX =
    (len_sunny_no / len_no) *
    (len_hot_no / len_no) *
    p_no

# normalize probabilities

p_yes_newX_n = p_yes_newX / (p_yes_newX + p_no_newX)

p_no_newX_n = p_no_newX / (p_yes_newX + p_no_newX)

# Prediction 2: newX = ["sunny", "cool", "high", true]

p_yes_newX = 
    (len_sunny_yes / len_yes) *
    (len_cool_yes / len_yes) *
    (len_high_yes / len_yes) *
    (len_windy_yes / len_yes) *
    p_yes
p_no_newX = 
    (len_sunny_no / len_no) *
    (len_cool_no / len_no) *
    (len_high_no / len_no) *
    (len_windy_no / len_no) *
    p_no

# normalize probabilities

p_yes_newX_n = p_yes_newX / (p_yes_newX + p_no_newX)

p_no_newX_n = p_no_newX / (p_yes_newX + p_no_newX)

# Prediction 3: newX = ["overcast", "mild", "high", false]

p_yes_newX =
    (len_overcast_yes / len_yes) *
    (len_mild_yes / len_yes) *
    (len_high_yes / len_yes) *
    (len_not_windy_yes / len_yes) *
    len_yes
p_no_newX =
    (len_overcast_no / len_no) *
    (len_mild_no / len_no) *
    (len_high_no / len_no) *
    (len_not_windy_no / len_no) *
    len_no

