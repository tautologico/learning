#
# xor_train.jl
# XOR test training a network with backprop
#
# Andrei de A. Formiga, 2012-05-29
#

load("mlpnnets.jl")

function build_xor_net()
    xornn = MLPNNet(2, [2, 1])
    srand(473199571347)
    random_weights(xornn, 1.75)

    xornn
end

function build_dataset()
    ds = DataSet(4, 2, 1)
    ds.data = [0.0 0.0; 0.0 0.9; 0.9 0.0; 0.9 0.9]
    ds.outs[1] = 0.0
    ds.outs[2] = 0.9
    ds.outs[3] = 0.9
    ds.outs[4] = 0.0
    ds
end

function test_xor(epochs, lrate)
    xornn = build_xor_net()
    ds = build_dataset()

    println("*** Start training")
    batch_train_bprop_vec(xornn, ds, epochs, lrate, sigmoid, dsigmoid)
    println("*** Training concluded")

    println("### Test: ")
    inputs = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]'

    for i = 1:size(inputs, 2)
        outs = present_input_vec(xornn, inputs[:,i], sigmoid)
        println("Outputs for $(inputs[:,i]) = $outs")                    
    end
end
