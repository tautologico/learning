
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

# constructor should verify compatibility between initvec and transmatrix
type MarkovChain
    initvec
    transmatrix
end

# some chains based on the above matrices
gothmc1 = MarkovChain([1.0  0.0], gothenburg_weather)  # start always at first state (sunny)
gothmc2 = MarkovChain([0.5  0.5], gothenburg_weather) 

lamc1 = MarkovChain([0.5  0.5], la_weather)

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
    dist = mc.initvec
    state = init(mc)
    res[1] = state == 1 ? 1 : 0
    for i = 2:steps
        state = simulation_step(mc, state)
        dist *= mc.transmatrix
        res[i] = res[i-1] + (state == 1 ? 1 : 0)
    end
    (res, dist)
end

# calculate the proportion of steps with state equal to 1 for each step, 
# given the cumulative number of steps with state = 1
state1_proportion(states) = states ./ [1:length(states)]

    
# create a grid for the hardcore model with n lines, m columns
create_hardcore_grid(n, m) = zeros(Uint8, n, m)

function print_grid(grid)
    for i = 1:size(grid, 1)-1
        for j = 1:size(grid, 2)-1
            print(grid[i, j] == 0 ? "O" : "X")
            print(" -- ")
        end
        println(grid[i, size(grid, 2)] == 0 ? "O" : "X")
        for j = 1:size(grid, 2)
            print("|    ")
        end
        println("")
    end
    for j = 1:size(grid, 2)-1
        print(grid[size(grid, 1), j] == 0 ? "O" : "X")
        print(" -- ")
    end
    println(grid[size(grid, 1), size(grid, 2)] == 0 ? "O" : "X")
end
    
function copy_neighbors(grid, i, j, neighbors)
    ix = 1
    if i-1 >= 1
        neighbors[ix] = grid[i-1, j]
        ix += 1
    end
    if j-1 >= 1
        neighbors[ix] = grid[i, j-1]
        ix += 1
    end
    if j+1 <= size(grid, 2)
        neighbors[ix] = grid[i, j+1]
        ix += 1
    end
    if i+1 <= size(grid, 1)
        neighbors[ix] = grid[i+1, j]
        ix += 1
    end
end

# returns a vector with the values of all neighbors of 
# the position at line i, column j
function neighbors(grid, i, j)
    w = (i > 1 && i < size(grid, 1) ? 2 : 1)
    h = (j > 1 && j < size(grid, 2) ? 2 : 1)
    res = zeros(eltype(grid), w + h)
    copy_neighbors(grid, i, j, res)
    res
end

function hardcore_mcmc(grid, steps)
    k = length(grid)
    delta = 1.0 / (2 * k)
    choices = [0.0:delta:1.0]
    for s = 1:steps
        r = rand()
        i = 1
        while choices[i] <= r && i < 2 * k
            i += 1
        end
        # check if heads or tails
        heads = (i % 2 == 0)
        v = div(i-1, 2) + 1
        ig = mod(v-1, size(grid, 1)) + 1
        jg = div(v-1, size(grid, 1)) + 1
        #println("i = $i, v = $v, ig = $ig, jg = $jg")
        if heads && max(neighbors(grid, ig, jg)) == 0
            grid[ig, jg] = 1
        else
            grid[ig, jg] = 0
        end
    end
end
