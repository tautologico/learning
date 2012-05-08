#
# xor_basic.jl
# Basic XOR test without training
#
# Andrei de A. Formiga, 2012-05-07
#

load("mlpnnets.jl")

function build_xor_net()
    xornn = MLPNNet(2, [2, 1])
    xornn.layers[1].weights[1, :] = [0.5  -1.0  -1.0]
    xornn.layers[1].weights[2, :] = [-1.5  1.0   1.0]
    xornn.layers[2].weights[1, :] = [0.5  -1.0  -1.0]

    xornn
end

function test_xor()
    xornn = build_xor_net()

    inputs = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]'
    expected_outs = [0.0, 1.0, 1.0, 0.0]

    for i = 1:size(inputs, 2)
        outs = present_input_vec(xornn, inputs[:,i], threshold)
        @assert outs[1] == expected_outs[i]
        println("Outputs for $(inputs[:,i]) = $outs")                    
    end
    println("All tests passed")    
end
    