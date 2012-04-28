
#
# mkvchains.jl
# Markov Chain simulation
#
# Andrei de A. Formiga, 2012-04-27
#

gothenburg_weather = [ 0.75  0.25 
                       0.25  0.75 ]

type MarkovChain
    initfn
    transmatrix
end

function simulate(mc::MarkovChain)
    println("Simulation")
end
