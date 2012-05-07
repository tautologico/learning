#
# mlpnnets.jl
# Feed-forward MLP neural network implementation
# Training with backpropagation
#
# Andrei de A. Formiga, 2012-05-07
#

type MLPLayer
    n_neurons::Int
    weights::Array{Float64,2}
    outputs::Vector{Float64}

    function MLPLayer(n_neurons::Int, n_neurons_prev::Int)
        new(n_neurons, zeros(Float64, n_neurons, n_neurons_prev+1), zeros(Float64, n_neurons))
    end
end

type MLPNNet
    n_inputs::Int
    layers::Vector{MLPLayer}

    function MLPNNet(n_inputs::Int, layers::Vector{Int})
        if length(layers) == 0
            error("MLPNNet: can't create network with 0 layers")
        end
        
        l = Array(MLPLayer, length(layers))
        l[1] = MLPLayer(layers[1], n_inputs)
        for i = 2:length(l)
            l[i] = MLPLayer(layers[i], layers[i-1])
        end
        new(n_inputs, l)
    end
end

get_outputs(nnet::MLPNNet) = nnet.layers[end].outputs

type DataSet
    n_cases::Int
    n_fields::Int
    data::Array{Float64, 2}

    DataSet(n_cases, n_fields) = new(n_cases, n_fields, zeros(Float64, n_cases, n_fields))
end

# initialize weights of the network randomly, with a maximum absolute value
function random_weights(nnet::MLPNNet, maxabs::Float64)
    for l in nnet.layers
        l.weights = map(x -> (rand() - 0.5) * maxabs / 2, l.weights)
    end
end

# present input to the neural network, calculating outputs for all layers
function present_input(nnet::MLPNNet, input::Vector{Float64})
end

# vectorized version of present_input
function present_input_vec(nnet::MLPNNet, input::Vector{Float64})
    nnet.layers[1].outputs = nnet.layers[1].weights * [1.0, inputs]
    for i in 2:length(nnet.layers)
        nnet.layers[i].outputs = nnet.layers[i].weights * [1.0, nnet.layers[i-1].outputs]
    end
    get_outputs(nnet)
end
