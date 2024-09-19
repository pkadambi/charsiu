import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8.5/1.5, 5/1.2))

bars = ['MFA w/ SAT', 'Wav2Vec2\nno SAT', 'Wav2Vec2-\nxVec (OURS)', 'Inter-Rater\nAgreement']
plt.rcParams.update({'font.size': 30})
yvals = [87.7, 91.5, 92.9, 93.7]
colors = ['r', 'k', 'g', 'b']
ax.bar(bars, yvals, color=colors, alpha=.6)

ax.set_title('Accuracy\nby Aligner', size=20, fontweight='bold')
ax.set_xlabel('Alignment Method', fontsize=16, fontweight='bold')
ax.set_ylabel('Alignment Accuracy (%)', fontsize=16, fontweight='bold')
ax.legend()
plt.ylim(85, 96)
ax.grid()

plt.text()
# plt.text()

plt.show()
