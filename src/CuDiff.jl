module CuDiff

sqr(x) = (x*x)
cube(x) = (x*x*x)
inverse(x) = one(x) / x

include("./dual.jl")
# include("./single.jl")
# include("./scalar.jl")

function derivative(f, x, p...)
    dual = f(Dual(x, one(x)), p...)
    return dual.u, dual.du
end

# function evaluate_gpu(f, x, p...)
#     single = f(Single(x), p...)
#     return single.u
# end
#
# function evaluate_cpu(f, x, p...)
#     scalar = f(Scalar(x), p...)
#     return scalar.u
# end

export derivative

end # module
