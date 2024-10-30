from surrogate_matching_simple import run_matching_simple
from surrogate_matching_perturbed import run_matching_perturbed


#Runs experiment many times to compare perturbed

simpleSumCos = 0
simpleSumAngle = 0
perturbedSumCos = 0
perturbedSumAngle = 0


for i in range(10):
    _, scos,sangle = run_matching_simple()
    simpleSumCos += scos
    simpleSumAngle += sangle
    _, pcos,pangle = run_matching_perturbed()
    perturbedSumCos += pcos
    perturbedSumAngle += pangle


simpleAvgCos = simpleSumCos/100
simpleAvgAngle = simpleSumAngle/100

perturbedAvgCos = perturbedSumCos/100
perturbedAvgAngle = perturbedSumAngle/100


print(simpleAvgCos)
print(simpleAvgAngle)
print(perturbedAvgCos)
print(perturbedAvgAngle)

