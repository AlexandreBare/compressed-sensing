using LinearAlgebra
using ImageView
using JuMP
using Gurobi
using MosekTools

include("utilities.jl")

dir = "$(pwd())/data/"
M = [608, 1014, 1521, 3042]
dim_image = (78, 78)
N = dim_image[1] * dim_image[2]
psi = unpickler("$(dir)basis_matrix.pickle")

a = 1
m = unpickler("$(dir)uncorrupted_measurements_M$(M[a]).pickle")
#m = unpickler("$(dir)noisy_measurements_M$(M[i]).pickle")
phi = unpickler("$(dir)measurement_matrix_M$(M[a]).pickle")
r_image = transpose(m) / transpose(phi)
image = reshape(r_image, dim_image)
imshow(image)
r = transpose(r_image)

#for a in 1:1#1:size(M)[1]
    #m = unpickler("$(dir)uncorrupted_measurements_M$(M[a]).pickle")
    #m = unpickler("$(dir)noisy_measurements_M$(M[i]).pickle")
    #phi = unpickler("$(dir)measurement_matrix_M$(M[a]).pickle")
    #r_image = transpose(m) / transpose(phi)
    #image = reshape(r_image, dim_image)
    #imshow(image)
#end

# CREATE EMPTY MODEL

LP_model = Model(Gurobi.Optimizer)

# ADD VARIABLES

@variable(LP_model, r_hat[1:N])
@variable(LP_model, r[1:N])
@variable(LP_model, m[1:M[a]])
@variable(LP_model, x_hat[1:N])
#@variable(LP_model, 0 <= t[1:N])
@variable(LP_model, 0 <= t)

# ADD CONSTRAINTS

#@constraint(LP_model, bounded_above_diff[j = 1:N], r_hat[j] - r[j] <= t[j])
#@constraint(LP_model, bounded_below_diff[j = 1:N], -t[j] <= r_hat[j] - r[j])
#@constraint(LP_model, bounded_above_diff[j = 1:N], r_hat[j] - r[j] <= t[j])
#@constraint(LP_model, bounded_below_diff[j = 1:N], -t[j] <= r_hat[j] - r[j])
@constraint(LP_model, bounded_above_diff, sum(r_hat - r) <= t)
@constraint(LP_model, bounded_below_diff, -t <= sum(r_hat - r))
@constraint(LP_model, true_representation[i = 1:M[a]], transpose(phi[i, :]) * r == m[i])
@constraint(LP_model, sparse_representation[j = 1:N], transpose(psi[:, j]) * x_hat == r_hat[j])

# ADD OBJECTIVE

#@objective(LP_model, Min, sum(t))
@objective(LP_model, Min, t)

# WRITE MODEL TO FILE

#write_to_file(LP_model, "model.mps") # Create files of more than 2GB

# SOLVE MODEL

optimize!(LP_model)

# RETRIEVE SOLUTIONS
x_h = value.(x_hat)
r_hh = value.(r_hat)

# Visualise reconsituted image
r_h = transpose(psi) * x_h
#print(value.(r_hat) - r_h)
_image = reshape(r_hh, dim_image)
imshow(_image)
