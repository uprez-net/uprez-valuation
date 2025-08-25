# Step 2: Analysis & Quantification(Without Prospectus)

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

This is where the major changes occur. The AI can no longer assume that all information will be in one place (a Prospectus). It must now act more like a human analyst, piecing together a conclusion from multiple pieces of evidence.

### **Insight 2.1: Growth & Narrative Score**

- **Meaningful Insight:** Unchanged. A score from -1.0 to +1.0 representing the strength of the company's growth story.
- **Required Data Points (Now More Complex):**
    - The entire collection of narrative_summary, growth_drivers, and management_highlights extracted from *all* uploaded strategic documents (Investor Decks, Business Plans, IMs).
    - historical_performance data (from Financial Statements).
- **Calculation & AI Processing Logic (The New Logic):**
    1. **Cross-Document Consistency Check:** This is a new, crucial step. The AI now checks for consistency across the different documents.
        - **Positive Signal:** Does the Investor Deck's growth plan align with the detailed strategy in the Business Plan? Consistent messaging increases the score.
        - **Negative Signal:** Does one document claim a 50% market share while another suggests 15%? Inconsistencies will significantly penalize the score and be flagged for user review.
    2. **Evidence-Based Scoring:** The model's scoring is now based on the **quality and consistency of evidence**.
        - A claim like "expand to the UK market" is given a higher score if it's mentioned in both the Board Strategy document *and* the Investor Presentation.
    3. **Historical Validation (Unchanged but more important):** The AI still cross-references growth claims with historical financial performance. A claim of being a "high-growth company" is now validated against the extracted historical CAGR. A significant mismatch between the narrative ("explosive growth") and the numbers (5% growth) will lower the score.
- **AI Model & Training (Now More Advanced):**
    - **Model:** This moves beyond simple text classification to a model capable of **cross-document analysis and reasoning**. This might involve techniques like:
        - **Knowledge Graphs:** Building a simple graph of claims and their sources to find connections and contradictions.
        - **Advanced RAG:** Using a vector database not just to find rules, but to find similar or contradictory statements across the user's own document set.
    - **Training:** The training dataset now needs to consist of *sets* of documents (e.g., a pitch deck, business plan, and financials for Company X) along with an expert-assigned overall score. The model learns what a "coherent" and "incoherent" set of strategic documents looks like.
- **Stored Output (Now includes evidence trail):**codeJSON
    
    ```
    {
      "growth_narrative_score": 0.65, // Good, but not perfect
      "justification": "Consistent growth story across Investor Deck and Business Plan, supported by a strong 40% historical revenue CAGR. Score was moderated due to a lack of specific timelines for new product launches.",
      "supporting_evidence": [
        {"claim": "UK Market Expansion", "sources": ["Investor_Deck_Q3.pptx", "Board_Minutes_May.pdf"]},
        {"claim": "40% CAGR", "sources": ["Audited_Financials_FY23.pdf"]}
      ],
      "conflicting_evidence": []
    }
    ```
    

### **Insight 2.2: Risk Weighting Factor**

- **Meaningful Insight:** Unchanged. A percentage discount based on the company's risk profile.
- **Required Data Points (Now from multiple sources):**
    - The entire collection of risks extracted from the Risk Register, Board Minutes, DDQs, etc.
    - audit_opinion_fy1 and audit_opinion_fy2.
- **Calculation & AI Processing Logic (The New Logic):**
    1. **Risk Aggregation & De-duplication:** The AI first aggregates all identified risks from all uploaded documents. It must be smart enough to recognize that "high dependency on our key supplier" in the Board Minutes is the same risk as "Supplier Concentration Risk" in the Risk Register, and not double-count it.
    2. **Mitigation-Adjusted Severity:** This is a crucial enhancement. After identifying a risk, the AI actively searches all documents for a corresponding mitigation strategy.
        - **Scenario A (No Mitigation):** A high-severity risk with no mentioned mitigation strategy receives the full penalty (e.g., -15%).
        - **Scenario B (Mitigation Found):** The same risk, but the AI finds a statement in the Board Minutes like "Plan to onboard a secondary supplier is on track for Q4," will have its penalty reduced (e.g., to -7%).
    3. **Weighted Summation:** The final risk factor is the sum of all the mitigation-adjusted risk penalties.
- **AI Model & Training:**
    - **Model:** This requires a more sophisticated NLP pipeline.
        1. **Risk Identification Model:** Identifies potential risk statements.
        2. **Risk-Mitigation Linking Model:** A model that can understand the relationship between a stated risk and a stated mitigation strategy, even if they are in different documents (e.g., using sentence embeddings to find semantic similarity).
    - **Training:** The model would be trained on a dataset where experts have highlighted both a risk statement and the corresponding mitigation statement(s) within a set of company documents. The model learns to connect these pairs.
- **Stored Output (Now includes mitigation status):**codeJSON
    
    ```
    {
      "risk_weighting_factor": -0.12, // A 12% discount
      "key_risks": [
        {
          "risk": "Single customer dependency (Severity: 0.9)",
          "mitigation_status": "Mitigated",
          "mitigation_evidence": "Board minutes confirm a new major customer was signed in Q3, reducing dependency from 60% to 35% of revenue.",
          "sources": ["Risk_Register.xlsx", "Board_Minutes_Aug.pdf"]
        }
      ]
    }
    ```
    

### **Summary of Key Changes for the Developer**

- **From Static to Dynamic Analysis:** The logic can't assume a fixed structure. It must be able to synthesize findings from a heterogeneous set of documents.
- **Emphasis on Cross-Verification:** A core feature of the new Step 2 is the AI's ability to check claims across documents and against historical data. This is a huge value-add.
- **AI Models Are More Complex:** We move from simple classification to models that understand context, relationships, and contradictions across multiple sources.
- **The Output is Richer:** The stored JSON is no longer just a score. It's a conclusion supported by a clear, traceable **evidence trail**, which is crucial for building trust with the expert user (Raj, the Lead Manager).