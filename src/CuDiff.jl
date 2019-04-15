module CuDiff

sqr(x) = (x*x)
cube(x) = (x*x*x)
inverse(x) = one(x) / x

include("./dual.jl")
# include("./single.jl")

function derivative(f, x, p...)
    dual = f(Dual(x, one(x)), p...)
    return dual.u, dual.du
end

function evaluate(f, x, p...)
    res = f(Single(x), p...)
    return map(x -> isa(Dual) ? x.u : x, res)
end

export derivative, evaluate

end # module
