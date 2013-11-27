
# A discrete type
type DType
    name::String
    card::Int
    values::Vector

    function DType(name::String, values::Vector)
        new(name, length(values), values)
    end
end

booltype = DType("bool", [true, false])
    
# A discrete variable
type DVar
    name::String
    typ::DType
end

function nthValue(t::DType, index::Int)
    t.values[index]
end

function valueIndex(t::DType, value)
    if contains(t.values, value)
        find(t.values .== value)[1]
    else
        nothing
    end
end

type Assignment
    vars::Vector{DVar}
    values::Vector{Int}

    Assignment(vars::Vector{DVar}, values::Vector{Int}) = new(vars, values)
end

type Factor
    vars::Vector{DVar}
    values::Vector{Float64}

    Factor(vars::Vector{DVar}, values::Vector{Float64}) = new(vars, values)
end

function indexOfAssignment(f::Factor, a::Assignment)
    ix = a.values[1]
    for i = 2:numel(a.vars)
        ix += a.values[i] * a.vars[i-1].typ.card
    end
    ix
end

function indexToAssignment(f::Factor, ix::Int)
    indices = zeros(Int, length(f.vars))
    i = length(indices)
    ix = ix - 1
    while i > 1
        #indices[i] = div(ix, f.vars[i-1].typ.card)
        #ix = ix % f.vars[i].typ.card
        i -= 1
    end
    Assignment(f.vars, indices)
end

function getValueOfAssignment(f::Factor, a::Assignment)
    f.values[indexOfAssigment(f, a)]
end

function setValueOfAssignment(f::Factor, a::Assignment, val::Float64)
end

## Tests
X = DVar("X", booltype)
Y = DVar("Y", booltype)

f1 = Factor([X, Y], [0.25, 0.25, 0.25, 0.25])


# type DVar{T}
#     card::Int
#     values::Vector{T}

#     function DVar{T}(vals::Vector{T})
#         new(length(vals), vals)
#     end
# end

# function nthValue{T}(v::DVar{T}, index::Int)
#     v.values[index]
# end

# function valueIndex{T}(v::DVar{T}, value::T)
#     if contains(v.values, value)
#         find(v.values .== value)[1]
#     else
#         nothing
#     end
# end

# # A factor that stores values in a vector ordered by the assignments
# #
# # TODO: what if a factor has (discrete) variables of many different types?
# #       possibility: store all variables as ints, associate them with a Type,
# #       store mapping from int to specific value of type if needed
# type OrderedDiscreteFactor{VT, FT}
#     vars::Vector{DVar{VT}}
#     values::Vector{FT}

#     function OrderedDiscreteFactor{VT, FT}(vars::Vector{DVar{DT}}, values::Vector{FT})
#         valLength = prod(map(v -> v.card, vars))
#         @assert valLength == length(values)
#         new(vars, values)
#     end
# end

# type Assignment{T}
#     vars::Vector{DVar{T}}
#     values::Vector{Int}

#     function Assignment{T}(vars::Vector{DVar{T}}, values::Vector{T})
#         @assert length(vars) == length(values)       
#     end

#     function Assignment{T}(vars::Vector{DVar{T}}, valIndices::Vector{Int})
#     end
# end

# function indexToAssignment{T}(vars::Vector{DVar{T}}, index::Int)
#     indices = zeros(Int, length(vars))
#     i = 1
#     index = index - 1
#     while index > 0
#         indices[i] = div(index, vars[i].card)
#         index = index % vars[i].card
#     end
#     #Assignment{T}(vars, indices)
#     indices
# end

# function set{VT,FT}(f::OrderedDiscreteFactor{VT,FT}, a::Assignment{VT}, val::VT)
# end
