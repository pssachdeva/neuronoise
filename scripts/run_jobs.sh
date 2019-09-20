for N in 8 10 12 14; do
	for sigmaM in 0.5 1.0; do
		sbatch mi.py $N $sigmaM
	done
done
