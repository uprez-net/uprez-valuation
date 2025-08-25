# Step 2: Analysis & Quantification(With Prospectus)

For each dimension, we will detail:

1. **The Meaningful Insight:** The specific analytical output we need to generate.
2. **Required Data Points:** The exact inputs from Step 1 needed for the calculation.
3. **Calculation & AI Processing Logic:** How the AI model will process the inputs to produce the output.
4. **Marking & Criteria Checks:** The scale or criteria used to score or classify the result.
5. **AI Model & Training:** The type of AI model needed and how it would be trained.
6. **Stored Output:** The final, structured JSON object that this step produces.

---

### **Valuation Engine - Step 2: Analysis & Quantification**

**Objective:** To analyze the ingested data from each dimension and produce a set of quantified scores, multiples, and adjustment factors that represent the company's valuation profile.

---

### **1. Industry Dimension Analysis**

**Purpose:** To establish an objective, market-based valuation benchmark.

### **Insight 1.1: Baseline Valuation Multiple**

- **Meaningful Insight:** This is the median valuation multiple (e.g., Price-to-Earnings ratio) of a group of publicly traded, comparable companies on the ASX. It represents the "fair market price" for a company in this sector *before* considering our specific company's unique strengths and weaknesses.
- **Required Data Points:**
    - peer_group_tickers (User input)
    - live_market_data for each peer (from Market Data API): Market Cap, Enterprise Value, TTM Revenue, TTM EBITDA, TTM Net Profit.
- **Calculation & AI Processing Logic:**
    1. For each peer company in the list, calculate the three key valuation ratios:
        - P/E Ratio = Market Cap / Net Profit
        - EV/EBITDA Ratio = Enterprise Value / EBITDA
        - EV/Sales Ratio = Enterprise Value / Sales
    2. Collect all the calculated ratios into three separate arrays (one for each ratio type).
    3. **Outlier Removal:** Discard the top and bottom 10-15% of the values in each array to remove extreme outliers that could skew the result.
    4. Calculate the **median** of the remaining values in each array. The median is used instead of the mean (average) as it is less sensitive to remaining outliers.
- **Marking & Criteria Checks:** This is a direct calculation, not a subjective score. The criteria is statistical validity (i.e., having a large enough peer set and removing outliers).
- **AI Model & Training:** This step is primarily **algorithmic and statistical**, not a deep learning task. No complex AI training is needed. The "intelligence" is in the logic of the outlier removal and the choice of median over mean.
- **Stored Output (Example):** code JSON
    
    downloadcontent_copyexpand_less
    
    ```
    {
      "baseline_multiples": {
        "pe_ratio_median": 22.5,
        "ev_ebitda_median": 11.8,
        "ev_sales_median": 4.2,
        "peer_count": 8
      }
    }
    ```
    

---

### **2. Company-Centric Dimension Analysis**

**Purpose:** To translate the specific, qualitative aspects of the IPO company into quantitative adjustment factors. This is where NLP and machine learning are most critical.

### **Insight 2.1: Growth & Narrative Score**

- **Meaningful Insight:** A score from -1.0 to +1.0 representing the strength, clarity, and credibility of the company's growth story as presented in its Prospectus. A high score justifies a higher valuation multiple.
- **Required Data Points:**
    - narrative_themes (from Prospectus text)
    - narrative_sentiment (from Prospectus text)
    - growth_drivers (from Prospectus text)
    - historical_performance (from Financial Statements)
- **Calculation & AI Processing Logic:**
    1. The AI model will analyze the sentiment and themes of the narrative. Positive sentiment (e.g., words like "leading," "innovative," "disruptive") increases the score.
    2. It will cross-reference the stated growth_drivers with the historical_performance. If the company claims "rapid growth" and the historical data shows a 50% CAGR (Compound Annual Growth Rate), this is a strong positive signal. If historical growth is flat, the claim is weak, which penalizes the score.
    3. The specificity of the growth drivers is key. Vague statements ("we will grow") are scored lower than specific, measurable plans ("we will launch Product X in the UK market in Q3, targeting 10,000 users").
- **Marking & Criteria Checks:**
    - **+0.7 to +1.0 (Excellent):** Strong positive sentiment, clear narrative, specific growth drivers backed by strong historical performance.
    - **+0.2 to +0.6 (Good):** Positive sentiment, but growth drivers are less specific or historical performance is moderate.
    - **0.2 to +0.1 (Neutral/Average):** Generic narrative, vague plans.
    - **1.0 to -0.3 (Weak):** Negative sentiment, contradictory statements, or growth claims unsupported by historical data.
- **AI Model & Training:**
    - **Model:** A custom-trained **text classification model** (e.g., a fine-tuned BERT or similar transformer model).
    - **Training:** The model needs to be trained on a dataset of hundreds of past ASX Prospectuses. Each prospectus would be manually labeled by a human finance expert with a "Growth & Narrative Score". The model learns the correlation between certain phrases, financial data patterns, and the expert-assigned score.
- **Stored Output (Example):** code JSON
    
    downloadcontent_copyexpand_less
    
    IGNORE_WHEN_COPYING_START
    
    IGNORE_WHEN_COPYING_END
    
    ```
    {
      "growth_narrative_score": 0.75, // Corresponds to an excellent story
      "justification": "Strong positive sentiment in narrative, with specific growth drivers (UK expansion) supported by a 45% historical revenue CAGR."
    }
    ```
    

### **Insight 2.2: Risk Weighting Factor**

- **Meaningful Insight:** A percentage discount to be applied to the valuation multiple, reflecting the severity and number of unmitigated risks.
- **Required Data Points:**
    - risks (classified list with severity scores from Prospectus)
    - audit_opinion_fy1 (from Financial Statements)
- **Calculation & AI Processing Logic:**
    1. The AI model uses a weighted formula based on the classified risks. A high-severity risk in a critical category (e.g., "Operational - single customer dependency") carries a much larger penalty than a low-severity "Market" risk.
    2. The presence of a **"qualified" audit opinion** would automatically apply a very high base penalty.
    3. The model aggregates the penalties from all identified risks into a single discount factor. For example, one high-severity risk (-15%) and two medium-severity risks (-5% each) might result in a total risk factor of -25%.
- **Marking & Criteria Checks:** The output is a direct percentage discount, e.g., -5% (low risk), -15% (medium risk), -30% (high risk).
- **AI Model & Training:**
    - **Model:** A **rules-based expert system combined with an NLP model**. The NLP model classifies and scores the risks from the text. The expert system then applies a pre-defined set of rules to convert those scores into a final percentage discount.
    - **Training:** The NLP component would be trained on thousands of "Risk Factor" sentences, labeled by legal and financial experts for category and severity. The expert system's rules (e.g., "IF risk category is 'Regulatory' AND severity is > 0.8 THEN penalty = -20%") are crafted by human experts.
- **Stored Output (Example):** code JSON
    
    downloadcontent_copyexpand_less
    
    IGNORE_WHEN_COPYING_START
    
    IGNORE_WHEN_COPYING_END
    
    ```
    {
      "risk_weighting_factor": -0.15, // A 15% discount
      "key_risks_identified": ["Single customer dependency (Severity: 0.9)", "IP litigation risk (Severity: 0.7)"]
    }
    ```
    

---

### **3. Market Dimension Analysis**

**Purpose:** To quantify the impact of market timing and external sentiment on the valuation.

### **Insight 3.1: Market Sentiment & Hype Adjustment Factor**

- **Meaningful Insight:** A combined percentage adjustment factor that reflects whether the current market conditions are favorable ("a tailwind") or unfavorable ("a headwind") for this specific type of IPO.
- **Required Data Points:**
    - vix_data (current value and trend)
    - macro_data (interest rate direction)
    - sector_momentum (hype score and trend)
- **Calculation & AI Processing Logic:**
    1. This is a weighted combination of the inputs.
    2. **Sentiment Component:** A low and falling VIX contributes positively (e.g., +5%). A high and rising VIX contributes negatively (e.g., -10%).
    3. **Macro Component:** A stable or falling interest rate environment is a slight positive. A rising interest rate environment is a negative.
    4. **Hype Component:** This has the highest weighting. A high hype_score in a rising trend for the company's sector can add a significant positive adjustment (e.g., +15% to +25%). If the sector is out of favor, this could be neutral or slightly negative.
    5. The AI aggregates these components into a single factor.
- **Marking & Criteria Checks:** The output is a direct percentage adjustment, e.g., +20% (strong tailwind), +5% (slight tailwind), -10% (headwind).
- **AI Model & Training:** This is primarily a **weighted scoring algorithm** defined by financial experts. The NLP model for sector_momentum (as described in Step 1) is the key AI component here. The algorithm itself doesn't require deep learning, but rather expert-defined logic (e.g., "Hype factor is 3x more important than the VIX factor").
- **Stored Output (Example):** code JSON
    
    downloadcontent_copyexpand_less
    
    IGNORE_WHEN_COPYING_START
    
    IGNORE_WHEN_COPYING_END
    
    ```
    {
      "market_adjustment_factor": 0.20, // A 20% uplift due to market conditions
      "drivers": ["Positive market sentiment (VIX is low)", "Company is in a high-hype 'AI' sector"]
    }
    ```