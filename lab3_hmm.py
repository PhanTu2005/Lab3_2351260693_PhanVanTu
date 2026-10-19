import numpy as np
import librosa
from hmmlearn.hmm import GaussianHMM
from pathlib import Path

def extract_feat(path, sr=16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = librosa.effects.preemphasis(y, coef=0.97)
    
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13,
        n_fft=512, win_length=int(0.025*sr),
        hop_length=int(0.010*sr), window='hamming',
        n_mels=26, fmin=20, fmax=sr/2
    )
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    F = np.vstack([mfcc, d1, d2]).T
    
    mu = F.mean(axis=0, keepdims=True)
    sd = F.std(axis=0, keepdims=True) + 1e-8
    return (F - mu) / sd

def make_lr_hmm(n_states=5, n_iter=80):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type='diag',
        n_iter=n_iter, tol=1e-3,
        min_covar=1e-3,
        init_params='mc', 
        params='mc', 
        random_state=0
    )
    start = np.zeros(n_states)
    start[0] = 1.0
    A = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        A[i, i] = 0.7
        A[i, i+1] = 0.3
    A[-1, -1] = 1.0
    model.startprob_ = start
    model.transmat_ = A
    return model

labels = ['mo', 'dong', 'len', 'xuong', 'dung']
models = {}

print("Bắt đầu đọc dữ liệu và huấn luyện mô hình...")
for label in labels:
    feats = []
    lengths = []
    for wav in sorted(Path('dataset/train', label).glob('*.wav')):
        F = extract_feat(str(wav))
        feats.append(F)
        lengths.append(len(F))
        
    if len(feats) == 0:
        print(f"  -> LỖI: Không có file wav nào cho từ '{label}'! Kiểm tra lại thư mục.")
        continue

    X = np.vstack(feats)
    model = make_lr_hmm(n_states=5)
    model.fit(X, lengths)
    models[label] = model
    print(f"  -> Từ '{label}': Huấn luyện xong (Converged = {model.monitor_.converged})")

print("Huấn luyện hoàn tất!")

import matplotlib.pyplot as plt

def recognize(path, models):
    F = extract_feat(path)
    scores = {label: model.score(F) for label, model in models.items()}
    pred = max(scores, key=scores.get)
    return pred, scores

print("\n--- BẮT ĐẦU TEST & VITERBI ALIGNMENT ---")
# Tự động lấy file đầu tiên trong thư mục test/mo để thử
test_files = list(Path('dataset/test/mo').glob('*.wav'))
if len(test_files) > 0:
    test_file = str(test_files[0])
    print(f"Đang test file: {test_file}")
    
    # Dự đoán
    pred, scores = recognize(test_file, models)
    print('Dự đoán (Prediction) =', pred)
    print("Bảng điểm Acoustic Score (càng lớn càng tốt):")
    for k, v in sorted(scores.items(), key=lambda z: z[1], reverse=True):
        print(f'  {k:6s}: {v:10.2f}')

    # Vẽ Viterbi alignment
    F = extract_feat(test_file)
    logp, states = models['mo'].decode(F, algorithm='viterbi')
    
    plt.figure()
    plt.step(range(len(states)), states, where='post')
    plt.xlabel('Frame')
    plt.ylabel('State')
    plt.title(f'Viterbi alignment: {test_file} vs model MO')
    plt.show()
else:
    print("LỖI: Không tìm thấy file wav nào trong thư mục dataset/test/mo/")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

print("\n--- BẮT ĐẦU ĐÁNH GIÁ TOÀN BỘ TẬP TEST ---")
true_y = []
pred_y = []

for label in labels:
    # Đọc tất cả file trong thư mục test của từng từ lệnh
    for wav in sorted(Path('dataset/test', label).glob('*.wav')):
        pred, _ = recognize(str(wav), models)
        true_y.append(label)
        pred_y.append(pred)

# Tính Accuracy
acc = accuracy_score(true_y, pred_y)
print(f'Độ chính xác (Accuracy): {acc * 100:.2f}%')

# Hiển thị Confusion Matrix
cm = confusion_matrix(true_y, pred_y, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Lab 3 HMM')
plt.show()