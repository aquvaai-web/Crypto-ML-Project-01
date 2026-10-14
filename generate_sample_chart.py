import matplotlib.pyplot as plt
import numpy as np

# Generate a synthetic candlestick chart for testing upload
np.random.seed(42)
dates = np.arange(30)
prices = 64000 + np.cumsum(np.random.normal(150, 400, 30))

fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0a0d14')
ax.set_facecolor('#121824')

# Draw simulated candlesticks
for i in range(len(prices)-1):
    open_p = prices[i]
    close_p = prices[i+1]
    high_p = max(open_p, close_p) + np.random.uniform(50, 200)
    low_p = min(open_p, close_p) - np.random.uniform(50, 200)
    color = '#10b981' if close_p >= open_p else '#ef4444'
    
    # Wick
    ax.plot([i, i], [low_p, high_p], color=color, linewidth=1.5)
    # Body
    ax.bar(i, abs(close_p - open_p), bottom=min(open_p, close_p), color=color, width=0.6)

ax.set_title("BTC/USDT 1D Candlestick Chart (Sample)", color='#f1f5f9', fontsize=14, pad=15)
ax.tick_params(colors='#94a3b8')
for spine in ax.spines.values():
    spine.set_color('#232e42')

plt.tight_layout()
plt.savefig("sample_chart.png", dpi=150)
print("[Sample] Generated sample_chart.png successfully.")
