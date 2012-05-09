#
# mlpnnets.jl
# Feed-forward MLP neural network implementation
# Training with backpropagation
#
# Andrei de A. Formiga, 2012-05-07
#

# weights are organized in a 2-dimensional matrix.
# each line in the matrix corresponds to one neuron 
# in the current layer, and contains all the weights 
# arriving in the neuron from the previous layer
type MLPLayer
    n_neurons::Int
    weights::Matrix{Float64}
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
    n_outputs::Int
    data::Matrix{Float64}
    outs::Matrix{Float64}

    DataSet(n_cases, n_fields, n_outputs) =
        new(n_cases, n_fields, zeros(Float64, n_cases, n_fields),
                               zeros(Float64, n_cases, n_outputs))
end

# activation functions
threshold(x::Float64) = x > 0.0? 1.0 : 0.0

# initialize weights of the network randomly, with a maximum absolute value
function random_weights(nnet::MLPNNet, maxabs::Float64)
    for l in nnet.layers
        l.weights = map(x -> (rand() - 0.5) * maxabs / 2, l.weights)
    end
end

# present input to the neural network, calculating outputs for all layers
function present_input(nnet::MLPNNet, input::Vector{Float64}, actf)
end

# vectorized version of present_input
function present_input_vec(nnet::MLPNNet, inputs::Vector{Float64}, actf)
    @assert length(inputs) == nnet.n_inputs
    nnet.layers[1].outputs = map(actf, nnet.layers[1].weights * [1.0, inputs])
    
    for i in 2:length(nnet.layers)
        nnet.layers[i].outputs = map(actf,
                                     nnet.layers[i].weights * [1.0, nnet.layers[i-1].outputs])
    end
    get_outputs(nnet)
end

function batch_train_bprop_vec(nnet::MLPNNet, train_set::DataSet, epochs::Int, 
                               lrate::Float64, actf, dactf)
    # create matrix for deltas
    deltas = Array(Vector{Float64}, length(nnet.layers))
    for i = 1:length(nnet.layers)
        deltas[i] = Array(Float64, nnet.layers[i].n_neurons)
    end
       
    # create derivatives matrix
    # FIX: weights are per layer
    deriv = similar(nnet.weights)

    for ep = 1:epochs
        # forward propagation
        outs = present_input_vec(nnet, train_set.data[i,:], actf)
        err = outs - train_set.outs[i,:]

        # deltas for output layer
        deltas[end] = - err .* map(dactf, outs)

        # propagate deltas backwards
        for l = length(nnet.layers)-1:-1:1
            deltas[l] = nnet.layers[l+1].weights[:, 2:]' * deltas[l+1]
        end

        # TODO: calculate derivatives
    end
end
