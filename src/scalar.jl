import CUDAnative

struct Scalar{T}
    u::T
end

import Base: promote_rule, convert
import Base.+, Base.-, Base.*, Base./, Base.^
import Base.>, Base.<, Base.>=, Base.<=, Base.==, Base.!=
import Base.isapprox

zero(::Scalar{T}) where T = Scalar(zero(T))
one(::Scalar{T}) where T = Scalar(one(T))
convert(::Type{Scalar{T}}, c::T) where T<:Number = Scalar(c)
promote_rule(::Type{Scalar{T}}, ::Type{T}) where T<:Number = Scalar{T}

(+)(x::Scalar) = Scalar(x.u)
(+)(x::Scalar{T}, y::Scalar{T}) where T = Scalar(x.u+y.u)
(+)(x::Scalar{T}, c::Number) where T = (+)(promote(x,T(c))...)
(+)(c::Number, y::Scalar{T}) where T = (+)(promote(T(c),y)...)

(-)(x::Scalar) = Scalar(-x.u)
(-)(x::Scalar{T}, y::Scalar{T}) where T = Scalar(x.u-y.u)
(-)(x::Scalar{T}, c::Number) where T = (-)(promote(x,T(c))...)
(-)(c::Number, y::Scalar{T}) where T = (-)(promote(T(c),y)...)

(*)(x::Scalar{T}, y::Scalar{T}) where T = Scalar(x.u*y.u)
(*)(x::Scalar{T}, c::Number) where T = (*)(promote(x,T(c))...)
(*)(c::Number, y::Scalar{T}) where T = (*)(promote(T(c),y)...)

(/)(x::Scalar{T}, y::Scalar{T}) where T = Scalar(x.u/y.u)
(/)(x::Scalar{T}, c::Number) where T = (/)(promote(x,T(c))...)
(/)(c::Number, y::Scalar{T}) where T = (/)(promote(T(c),y)...)

CUDAnative.sin(x::Scalar) = Scalar(sin(x.u))
CUDAnative.sin_fast(x::Scalar) = Scalar(sin(x.u))
CUDAnative.sinpi(x::Scalar) = Scalar(sinpi(x.u))

CUDAnative.cos(x::Scalar) = Scalar(cos(x.u))
CUDAnative.cos_fast(x::Scalar) = Scalar(cos(x.u))
CUDAnative.cospi(x::Scalar) = Scalar(cospi(x.u))

CUDAnative.tan(x::Scalar) = Scalar(tan(x.u))
CUDAnative.tan_fast(x::Scalar) = Scalar(tan(x.u))

CUDAnative.acos(x::Scalar) = Scalar(acos(x.u))
CUDAnative.asin(x::Scalar) = Scalar(asin(x.u))
CUDAnative.atan(x::Scalar) = Scalar(atan(x.u))

CUDAnative.atan2(x::Scalar, y::Scalar) = Scalar(atan2(x.u,y.u))
CUDAnative.atan2(x::Scalar{T}, c::Number) where T = atan2(promote(x,T(c))...)
CUDAnative.atan2(c::Number, x::Scalar{T}) where T = atan2(promote(T(c),x)...)

CUDAnative.cosh(x::Scalar) = Scalar(cosh(x.u))
CUDAnative.sinh(x::Scalar) = Scalar(sinh(x.u))
CUDAnative.tanh(x::Scalar) = Scalar(tanh(x.u))

CUDAnative.acosh(x::Scalar) = Scalar(acosh(x.u))
CUDAnative.asinh(x::Scalar) = Scalar(asinh(x.u))
CUDAnative.atanh(x::Scalar) = Scalar(atanh(x.u))

CUDAnative.exp(x::Scalar) = Scalar(exp(x.u))
CUDAnative.exp_fast(x::Scalar) = Scalar(exp(x.u))
CUDAnative.exp2(x::Scalar) = Scalar(exp2(x.u))
CUDAnative.exp10(x::Scalar) = Scalar(exp10(x.u))
CUDAnative.expm1(x::Scalar) = Scalar(expm1(x.u))

CUDAnative.log(x::Scalar) = Scalar(log(x.u))
CUDAnative.log_fast(x::Scalar) = Scalar(log(x.u))
CUDAnative.log10(x::Scalar) = Scalar(log10(x.u))
CUDAnative.log10_fast(x::Scalar) = Scalar(log10(x.u))
CUDAnative.log2(x::Scalar) = Scalar(log2(x.u))
CUDAnative.log2_fast(x::Scalar) = Scalar(log2(x.u))
CUDAnative.log1p(x::Scalar) = Scalar(log1p(x.u))

CUDAnative.abs(x::Scalar) = Scalar(abs(x.u))
CUDAnative.saturate(x::Scalar) = Scalar(saturate(x.u))
CUDAnative.sqrt(x::Scalar) = Scalar(sqrt(x.u))
CUDAnative.rsqrt(x::Scalar) = Scalar(rsqrt(x.u))
CUDAnative.cbrt(x::Scalar) = Scalar(cbrt(x.u))
CUDAnative.rcbrt(x::Scalar) = Scalar(rcbrt(x.u))

CUDAnative.hypot(x::Scalar{T}, y::Scalar{T}) where T = Scalar(hypot(x.u, y.u))
CUDAnative.hypot(x::Scalar{T}, c::Number) where T = hypot(promote(x,T(c))...)
CUDAnative.hypot(c::Number, x::Scalar{T}) where T = hypot(promote(T(c),x)...)

CUDAnative.erf(x::Scalar) = Scalar(erf(x.u))
CUDAnative.erfc(x::Scalar) = Scalar(erfc(x.u))
CUDAnative.erfinv(x::Scalar) = Scalar(erfinv(x.u))
CUDAnative.erfcinv(x::Scalar) = Scalar(erfcinv(x.u))
CUDAnative.pow(x::Scalar, p) = Scalar(pow(x.u, p))
(^)(x::Scalar, p) = Scalar(x.u ^ p)

#############################################################################

CUDAnative.ceil(x::Scalar) = Scalar(ceil(x.u))
CUDAnative.floor(x::Scalar) = Scalar(floor(x.u))

CUDAnative.min(x::Scalar{T}, y::Scalar{T}) where T = min(x.u, y.u)
CUDAnative.min(x::Scalar{T}, c::Number) where T = min(x.u, c)
CUDAnative.min(c::Number, x::Scalar{T}) where T = min(c, x.u)
CUDAnative.max(x::Scalar{T}, y::Scalar{T}) where T = max(x.u, y.u)
CUDAnative.max(x::Scalar{T}, c::Number) where T = max(x.u, c)
CUDAnative.max(c::Number, x::Scalar{T}) where T = max(c, x.u)
CUDAnative.isfinite(x::Scalar) = isfinite(x.u)
CUDAnative.isinf(x::Scalar) = isinf(x.u)
CUDAnative.isnan(x::Scalar) = isnan(x.u)
CUDAnative.nearbyint(x::Scalar) = nearbyint(x.u)
CUDAnative.nextafter(x::Scalar) = nextafter(x.u)
CUDAnative.signbit(x::Scalar) = signbit(x.u)

(<)(x::Scalar{T}, y::Scalar{T}) where T = (x.u < y.u)
(<)(x::Scalar{T}, c::Number) where T = (x.u < c)
(<)(c::Number, x::Scalar{T}) where T = (c < x.u)

(>)(x::Scalar{T}, y::Scalar{T}) where T = (x.u > y.u)
(>)(x::Scalar{T}, c::Number) where T = (x.u > c)
(>)(c::Number, x::Scalar{T}) where T = (c > x.u)

(<=)(x::Scalar{T}, y::Scalar{T}) where T = (x.u <= y.u)
(<=)(x::Scalar{T}, c::Number) where T = (x.u <= c)
(<=)(c::Number, x::Scalar{T}) where T = (c <= x.u)

(>=)(x::Scalar{T}, y::Scalar{T}) where T = (x.u >= y.u)
(>=)(x::Scalar{T}, c::Number) where T = (x.u >= c)
(>=)(c::Number, x::Scalar{T}) where T = (c >= x.u)

(==)(x::Scalar{T}, y::Scalar{T}) where T = (x.u == y.u)
(==)(x::Scalar{T}, c::Number) where T = (x.u == c)
(==)(c::Number, x::Scalar{T}) where T = (c == x.u)

(!=)(x::Scalar{T}, y::Scalar{T}) where T = (x.u != y.u)
(!=)(x::Scalar{T}, c::Number) where T = (x.u != c)
(!=)(c::Number, x::Scalar{T}) where T = (c != x.u)

isapprox(x::Scalar{T}, y::Scalar{T}) where T = isapprox(x.u, y.u)
isapprox(x::Scalar{T}, c::Number) where T = isapprox(x.u, c)
isapprox(c::Number, x::Scalar{T}) where T = isapprox(c, x.u)
