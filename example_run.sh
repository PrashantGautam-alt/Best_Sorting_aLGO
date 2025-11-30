# 1. Run Benchmark
echo "--- Running Benchmark ---"
python3 src/alg_perf/benchmark.py

# 2. Train Model
echo "--- Training Model ---"
python3 src/alg_perf/train_model.py

# 3. Predict on a random list
echo "--- Predicting Best Algo ---"
python3 src/alg_perf/predict_best.py