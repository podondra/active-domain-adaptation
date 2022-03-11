include("ADA.jl")
using .ADA

traindigits!(MCDropoutLeNet(20), "mcdropoutlenet")
