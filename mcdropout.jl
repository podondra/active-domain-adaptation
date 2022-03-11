include("ADA.jl")
using .ADA

# TODO: GC.gc()
digits_al(mcdropout_sampling, MCDropoutLeNet, ("models/mcdropoutlenet.bson", 20), "data/mcdropout.csv")
