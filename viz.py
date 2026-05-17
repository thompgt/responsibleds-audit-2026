import matplotlib.pyplot as plt

regions = ['Europe (EU)', 'South America', 'Africa', 'Asia', 'North America']
bias_scores = [2.5, -3.2, -5.8, -4.1, -1.5] # In millions

plt.figure(figsize=(10, 6))
plt.bar(regions, bias_scores, color=['#4caf50' if s > 0 else '#f44336' for s in bias_scores])
plt.axhline(0, color='black', linewidth=1)
plt.title('Random Forest ADS: Valuation Bias by Player Origin Region')
plt.ylabel('Mean Valuation Error (Millions €)')
plt.xlabel('Player Origin Region')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('fairness_audit.png', bbox_inches='tight')
