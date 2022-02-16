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
a = 4
m = unpickler("$(dir)uncorrupted_measurements_M$(M[a]).pickle")
phi = unpickler("$(dir)measurement_matrix_M$(M[a]).pickle")

# Initial Image
r_image = transpose(m) / transpose(phi)
image = reshape(r_image, dim_image)
imshow(image)

# CREATE EMPTY MODEL

LP_model = Model(Gurobi.Optimizer)

# ADD VARIABLES

@variable(LP_model, x[1:N])
@variable(LP_model, 0 <= t)

# ADD CONSTRAINTS

@constraint(LP_model, second_order_cone, vcat(t, x) in SecondOrderCone())
@constraint(LP_model, signal_reconstruction, m .== phi * psi * x)

# ADD OBJECTIVE

@objective(LP_model, Min, t)

# SOLVE MODEL

optimize!(LP_model)

# RETRIEVE SOLUTIONS

x_hat = value.(x)
r_hat = psi * x_hat

# Final Image
_image = reshape(r_hat, dim_image)
imshow(_image)
