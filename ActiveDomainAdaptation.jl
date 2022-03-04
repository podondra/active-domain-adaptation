module ActiveDomainAdaptation

export simulate_al
export entropy_sampling, random_sampling, mcdropout_sampling
export human_labeller, oracle
export entropy, mc_mutual_information
export log2entropy

export LeNetVariant, MCLeNetVariant
export earlystopping!, finetune!, train!
export forward, predict, probability
export accuracy

include("activelearning.jl")
include("convnets.jl")

end
