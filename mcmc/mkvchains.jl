
#
# mkvchains.jl
# Markov Chain simulation
#
# Andrei de A. Formiga, 2012-04-27
#

gothenburg_weather = [ 0.75  0.25 
                       0.25  0.75 ]

la_weather = [ 0.5  0.5 
               0.1  0.9 ]

type MarkovChain
    initvec
    transmatrix
end

# some chains based on the above matrices
gothmc1 = MarkovChain([1.0, 0.0], gothenburg_weather)  # start always at first state (sunny)
gothmc2 = MarkovChain([0.5, 0.5], gothenburg_weather) 

lamc1 = MarkovChain([0.5, 0.5], la_weather)

# return the cumulative sum of a, because cumsum() sometimes
# returns a tuple
function cumulative_sum(a)
    res = zeros(eltype(a), length(a))
    res[1] = a[1]
    for i = 2:length(a)
        res[i] = res[i-1] + a[i]
    end
    res
end

function select_state(statevec)
    r = rand()
    trans = cumulative_sum(statevec)
    s = 1
    while trans[s] <= r
        s += 1
    end
    s
end

function init(mc::MarkovChain)
    select_state(mc.initvec)
end

function simulation_step(mc::MarkovChain, state)
    select_state(mc.transmatrix[state, :])
end

# simulate the Markov chain for the given number of steps
# return the number of steps with state = 1 for each step
function simulate(mc::MarkovChain, steps::Int)
    res = zeros(Int, steps)
    state = init(mc)
    res[1] = state == 1 ? 1 : 0
    for i = 2:steps
        state = simulation_step(mc, state)
        res[i] = res[i-1] + (state == 1 ? 1 : 0)
    end
    res
end

# calculate the proportion of steps with state equal to 1 for each step, 
# given the cumulative number of steps with state = 1
s1proportion(states) = states ./ [1:length(states)]

    