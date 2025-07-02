import pickle
import numpy as np

# ==== 1. 模型路径 ====
model_path = "./fungi_LR_model.pkl"

# ==== 2. 加载模型 ====
with open(model_path, "rb") as f:
    model = pickle.load(f)

# ==== 3. 输入分数 ====
# 示例分数，替换为你自己的
score1 = 0.590
score2 = 0.79
score3 = 0.91
input_scores = np.array([[score1, score2, score3]])

# ==== 4. 直接预测概率 ====
predicted_prob = model.predict_proba(input_scores)[0][1]

# ==== 5. 输出结果 ====
print(f"Input Scores: Score1={score1}, Score2={score2}, Score3={score3}")
print(f"Predicted Probability (Final Score): {predicted_prob:.4f}")
