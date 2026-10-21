using NeuralLinearSolve
using SparseArrays
using LinearAlgebra
using JLD2
using Test

plain_model_state(x::Union{Tuple, NamedTuple}) = all(plain_model_state, x)
plain_model_state(x::Array{Float32}) = true
plain_model_state(x::Union{Number, Symbol, Nothing}) = true
plain_model_state(x) = false

@testset "bundled model checkpoint" begin
    checkpoint = JLD2.load(
        joinpath(pkgdir(NeuralLinearSolve), "artifacts", "solver_model_cnn.jld2")
    )
    @test Set(keys(checkpoint)) == Set(["model_state", "label_names"])
    @test plain_model_state(checkpoint["model_state"])
    @test checkpoint["label_names"] == ["UMFPACK", "KLU", "Pardiso"]
end

@testset "frozen model predictions" begin
    rows = [i for i in 1:96 for j in 1:128 if mod(7i + 11j, 31) < 2]
    cols = [j for i in 1:96 for j in 1:128 if mod(7i + 11j, 31) < 2]
    cases = (
        spzeros(Float32, 64, 64),
        spdiagm(0 => ones(Float32, 64)),
        sparse(rows, cols, ones(Float32, length(rows)), 96, 128),
        sparse(ones(Float32, 64, 64)),
    )
    reference = Float32[
        0.3963515 2.0089237f-8 0.104794316 0.540196
        0.28280318 1.0 0.89366525 0.12125448
        0.3208453 6.9452005f-12 0.0015403226 0.33854946
    ]
    labels = (:UMFPACK, :KLU, :Pardiso)
    for (A, expected) in zip(cases, eachcol(reference))
        probs = predict_solver_probs(A)
        @test isapprox([probs[label] for label in labels], expected; rtol = 1.0f-5, atol = 1.0f-7)
        @test predict_solver(A) === labels[argmax(expected)]
    end
end

@testset "predict_solver returns valid symbol" begin
    A = sprand(1000, 1000, 0.01)
    result = predict_solver(A)
    @test result isa Symbol
    @test result in (:UMFPACK, :KLU, :Pardiso)
end

@testset "predict_solver_probs returns valid probabilities" begin
    A = sprand(1000, 1000, 0.01)
    probs = predict_solver_probs(A)
    @test probs isa Dict{Symbol, Float32}
    @test isapprox(sum(values(probs)), 1.0, atol = 1.0e-5)
    @test all(v >= 0 for v in values(probs))
end

@testset "diagonal matrix prediction" begin
    B = sparse(Diagonal(rand(500)))
    result = predict_solver(B)
    @test result isa Symbol
    @test result in (:UMFPACK, :KLU, :Pardiso)
end

@testset "spy plot generation" begin
    A = sprand(100, 100, 0.05)
    X = NeuralLinearSolve.matrix_to_spy(A)
    @test size(X) == (64, 64, 1, 1)
    @test eltype(X) == Float32
    @test all(0 .<= X .<= 1)
end
