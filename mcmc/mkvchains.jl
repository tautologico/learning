
#
# mkvchains.jl
# Markov Chain simulation
#
# Andrei de A. Formiga, 2012-04-27
#

type MarkovChain
    initfn
    transmatrix
end

function simulate(MarkovChain mc)
    println("Simulation")
end
