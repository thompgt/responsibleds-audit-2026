#  Technical Audit of an Algorithmic Decision-Support System

## Football Transfer Fee Analysis

This repository presents a technical audit of an existing Algorithmic Decision-Support System (ADS) that utilizes a Random Forest model to predict European soccer player valuations. While the baseline notebook optimizes purely for historical accuracy, this project investigates potential geographic and systemic biases embedded in the global transfer market training data. We extend the original analysis by rigorously evaluating the model against fairness metrics across key demographic axes, specifically focusing on player nationality and current league. By quantifying the tension between raw predictive precision and equitable valuation, the audit highlights areas of potential disparate impact within the algorithm. Ultimately, this framework provides actionable insights for developing more objective, transparent, and fair player assessment tools in sports analytics.

## Main Audit Takeaways

### 1. Fairness-Performance Frontier
The audit evaluates several model variants to identify the "sweet spot" between predictive accuracy and demographic fairness. As shown below, the weighted and conservative models significantly reduce the error gap between geographic regions while maintaining high overall performance.

![Model Comparison](./docs/assets/model_comparison.png)

### 2. Simpson's Paradox Diagnostics
A critical part of the audit is detecting Simpson's Paradox—where global trends are reversed in subgroups. The heatmap below identifies specific strata (like predicted fee buckets) where bias patterns may be hidden or misleading, ensuring a deeper level of granular fairness.

![Simpson's Paradox](./docs/assets/simpsons_paradox.png)

### 3. Model Explainability (SHAP)
Transparency is key to a responsible audit. We provide two distinct SHAP (SHapley Additive exPlanations) views to distinguish between **Market-Driven** and **Technical-Driven** valuations.

#### Global Importance (Market + Technical)
In the full model, we observe that technical skills (FIFA Overall/Potential) are primary drivers, but market-level features like the player's current market value also play a dominant role. While this provides high predictive accuracy, it potentially inherits historical biases embedded in market sentiment.

![SHAP Market](./docs/assets/shap_market.png)

#### 10-Factor Global Explainability (SHAP)
The plot below illustrates the global feature influence for our refined 10-factor model, where feature values are color-coded (red for high, blue for low) to show their directional impact on the transfer fee. Unlike traditional feature importance plots that only show magnitude, this SHAP beeswarm plot reveals how specific factors like high `Ability_Overall` or `Financial_Strength` consistently drive valuations upward, while factors like increased `Age_Feature` exert downward pressure. This directional transparency addresses the flaws of prior importance rankings by explicitly showing *how* a feature changes the outcome, rather than just *that* it does. From a fairness perspective, it allows us to audit whether sensitive proxies like `Passport_Premium` or `Home_Nation_Transfer` are exerting undue influence compared to intrinsic technical metrics like the `xG_Proxy`. By moving beyond "black-box" importance to granular directional impact, we provide a more transparent and auditable framework that ensures valuations are driven by performance and context rather than opaque systemic biases.

![SHAP 10 Factors](./plots/shap_summary.png)

### 4. Domain Expertise: LIME Scenario Analysis
Local Interpretable Model-agnostic Explanations (LIME) were used to conduct deep-dive case studies into specific player archetypes (e.g., young prospects from emerging regions vs. veterans in top leagues). These scenario-based audits allow domain experts to validate whether the model's local reasoning aligns with football scouting logic.

![LIME Scenarios](./docs/assets/lime_scenario_bars.png)

The summary below compares the model's sensitivity across different transfer scenarios, ensuring that valuation risks are understood at an individual level.

![LIME Summary](./docs/assets/lime_scenario_summary.png)

### 5. Fair 90% Conformal Prediction
To account for uncertainty in a responsible way, the audit implements **Conformal Prediction**. This moves beyond point estimates to provide a calibrated 90% confidence interval for each player's fee. The analysis confirms that these intervals maintain consistent coverage across different regions, providing a reliable measure of "valuation risk" that doesn't penalize players based on their origin.

![Conformal Prediction](./docs/assets/conformal_prediction.png)

### 6. Counterfactual Fairness (Shadow Model)
We conducted a counterfactual audit using a "Shadow Model" to measure the **Geographic Premium**. By simulating a scenario where a player's region is swapped while keeping their performance stats identical, we quantified the systemic bias present in the training data. This insight allows us to calibrate our fairness-aware model to be truly blind to these historical biases.

![Counterfactual Fairness](./docs/assets/counterfactual_fairness.png)
