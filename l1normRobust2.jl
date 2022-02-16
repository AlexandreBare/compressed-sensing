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
m = unpickler("$(dir)noisy_measurements_M$(M[a]).pickle")
#m = unpickler("$(dir)uncorrupted_measurements_M$(M[a]).pickle")
phi = unpickler("$(dir)measurement_matrix_M$(M[a]).pickle")

tolerance = 0.4

# Initial Image
r_image = transpose(m) / transpose(phi)
image = reshape(r_image, dim_image)
imshow(image)

# CREATE EMPTY MODEL

LP_model = Model(Gurobi.Optimizer)

# ADD VARIABLES

@variable(LP_model, x[1:N])
@variable(LP_model, 0 <= t[1:N])

# ADD CONSTRAINTS

@constraint(LP_model, x_bounded_below[j = 1:N], x[j] >= -t[j])
@constraint(LP_model, x_bounded_above[j = 1:N], x[j] <= t[j])
@constraint(LP_model, signal_reconstruction, vcat(tolerance, phi * psi * x - m) in SecondOrderCone())


# ADD OBJECTIVE

@objective(LP_model, Min, sum(t))

# SOLVE MODEL

optimize!(LP_model)

# RETRIEVE SOLUTIONS

x_hat = value.(x)
r_hat = psi * x_hat

# Final Image
_image = reshape(r_hat, dim_image)
imshow(_image)
