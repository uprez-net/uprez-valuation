# Step 3: Synthesis & Insight Generation

**Objective:** To take the quantified, analytical outputs from Step 2 (the baseline multiples and adjustment factors) and the key financial projections from Step 1, and synthesize them into a clear, actionable set of valuation insights for the user. This is the final calculation layer of the engine.

---

### **Insight 3.1: The Target Valuation Multiple**

- **Meaningful Insight:** This is the core valuation multiple (e.g., P/E ratio) that the platform has calculated as being appropriate for this specific company, in the current market, after accounting for all its unique strengths and weaknesses. It's the "justified" multiple.
- **Required Data Points (Inputs from Previous Steps):**
    - baseline_multiples (from Step 2, Insight 1.1) - The objective starting point.
    - growth_narrative_score (from Step 2, Insight 2.1) - The company-specific story strength.
    - risk_weighting_factor (from Step 2, Insight 2.2) - The company-specific risk discount.
    - market_adjustment_factor (from Step 2, Insight 3.1) - The external market timing factor.
- **Actions Performed (Calculation & Analysis Logic):**
    1. **Convert Scores to Factors:** The growth_narrative_score (which is on a -1.0 to +1.0 scale) needs to be converted into a multiplicative factor. The logic for this is an expert-defined business rule. A simple but effective rule would be:
        - Growth Narrative Factor = 1 + (growth_narrative_score * 0.25)
        - *Example:* A score of 0.8 becomes a factor of 1.20 (a 20% premium). A score of -0.4 becomes a factor of 0.90 (a 10% discount).
    2. **Apply All Factors:** The platform applies all the adjustment factors from Step 2 to the baseline multiple. The calculation is a clear, multi-step process:
        - Adjusted Multiple = baseline_multiples.pe_ratio_median * Growth Narrative Factor
        - Risk-Adjusted Multiple = Adjusted Multiple * (1 + risk_weighting_factor) *(Note: risk factor is negative, so this reduces the multiple)*
        - Final Target Multiple = Risk-Adjusted Multiple * (1 + market_adjustment_factor)
- **Criteria & Marking:** This is a calculated output. The criteria for its validity depend entirely on the traceability and justification of the input factors from Step 2. The UI should allow the user to see exactly how the baseline multiple was adjusted at each stage.
- **Stored Output (Example):**codeJSON
    
    ```
    {
      "target_multiple": {
        "type": "P/E Ratio",
        "value": 26.5,
        "calculation_trace": {
          "baseline_pe_median": 22.5,
          "growth_narrative_factor": 1.1875, // from score of 0.75
          "risk_weighting_factor": -0.15,
          "market_adjustment_factor": 0.20,
          "final_calculation": "22.5 * 1.1875 * (1 - 0.15) * (1 + 0.20) = 26.5"
        }
      }
    }
    ```
    

---

### **Insight 3.2: The Indicative Valuation Range**

- **Meaningful Insight:** The estimated total value of the company's equity (its market capitalisation) if it were to list today, presented as a range to reflect uncertainty.
- **Required Data Points (Inputs from Previous Steps):**
    - projected_npat (from Step 1, Financial Model) - The core earnings forecast.
    - target_multiple.value (from Step 3, Insight 3.1) - The final calculated multiple.
- **Actions Performed (Calculation & Analysis Logic):**
    1. **Calculate Central Valuation:** The primary calculation is straightforward:
        - Central Valuation = projected_npat * target_multiple.value
    2. **Generate a Range:** A single valuation number is misleadingly precise. The platform must generate a range. This is another expert-defined rule. A common approach is to create a +/- 10-15% range around the central value.
        - Low End Valuation = Central Valuation * 0.85
        - High End Valuation = Central Valuation * 1.15
- **Criteria & Marking:** The output is a monetary range. Its credibility is derived from the target_multiple calculation.
- **Stored Output (Example):**codeJSON
    
    ```
    {
      "indicative_valuation_range": {
        "low_end_aud": 71500000,
        "central_aud": 84125000, // (3M * 28.04 from example above)
        "high_end_aud": 96750000,
        "currency": "AUD"
      }
    }
    ```
    

---

### **Insight 3.3: Sweet Spot for Offering Price Per Share**

- **Meaningful Insight:** The final, user-facing output. It translates the total company valuation into a price for a single share, which is what is actually marketed to investors.
- **Required Data Points (Inputs from Previous Steps):**
    - indicative_valuation_range (from Step 3, Insight 3.2) - The calculated value of the whole company.
    - post_ipo_shares (from Step 1, extracted from Prospectus or calculated from user input) - The total number of shares that will exist after the IPO.
- **Actions Performed (Calculation & Analysis Logic):**
    1. The platform simply divides each point of the valuation range by the total number of post-IPO shares.
        - Low Price = indicative_valuation_range.low_end_aud / post_ipo_shares
        - High Price = indicative_valuation_range.high_end_aud / post_ipo_shares
- **Criteria & Marking:** This is the final calculated output. The UI should present this clearly, often rounded to the nearest cent.
- **Stored Output (Example):**codeJSON
    
    ```
    {
      "offering_price_range": {
        "low_price_aud": 1.43, // (71.5M / 50M shares)
        "high_price_aud": 1.94, // (96.75M / 50M shares)
        "currency": "AUD"
      }
    }
    ```
    

---

### **Insight 3.4: Potential Capital Raise Analysis**

- **Meaningful Insight:** An analysis of how much money the company will raise and what the implied valuation is, based on the offer price. This is crucial for the company to understand dilution and ensure their capital needs are met.
- **Required Data Points (Inputs from Previous Steps):**
    - offering_price_range (from Step 3, Insight 3.3)
    - deal_assumptions.capital_raise (from Step 1, User Input if no Prospectus) OR data extracted from the "Details of the Offer" section in the Prospectus.
- **Actions Performed (Calculation & Analysis Logic):**
    1. The platform presents a scenario table. It allows the user to see the relationship between the capital they want to raise and the number of shares they need to issue at a given price.
    2. Shares to Issue = Target Capital Raise / Issue Price
    3. Post-IPO Valuation = Issue Price * Total Post-IPO Shares
- **Criteria & Marking:** This is an interactive modeling tool rather than a static output. The UI will feature a table that updates in real-time as the user adjusts variables.
- **Stored Output (Example):** The output is a set of scenarios presented to the user.codeJSON
    
    ```
    {
      "capital_raise_scenarios": [
        {
          "issue_price": 1.50,
          "shares_to_issue_for_20M": 13333333,
          "implied_post_money_valuation": 80000000
        },
        {
          "issue_price": 1.75,
          "shares_to_issue_for_20M": 11428571,
          "implied_post_money_valuation": 93333333
        }
      ]
    }
    ```
    

---

### **Changes for the "Without Prospectus" Scenario**

The logic and calculations in **Step 3 do not change at all**. This is the key insight.

The **inputs** to Step 3, however, are derived differently in the "Without Prospectus" world (as defined in our redefined Step 1 and 2):

- The growth_narrative_score and risk_weighting_factor from Step 2 will be based on a **synthesis of multiple internal documents** (Pitch Decks, Business Plans, Risk Registers) instead of just the Prospectus. They may have a slightly lower confidence score attached to them, which could be reflected in a slightly wider final valuation range (e.g., +/- 20% instead of +/- 15%).
- The post_ipo_shares data point, which is critical for the Offering Price Per Share calculation, is no longer a hard fact extracted from the Prospectus. It is now **calculated based on the user's direct input** regarding their desired capital raise and sell-down percentage (deal_assumptions from Step 1).

**In summary for the developer:** The core calculation engine of Step 3 is agnostic to the source of the data. As long as Step 1 and Step 2 provide it with the required quantified inputs (baseline_multiples, adjustment_factors, projected_npat, post_ipo_shares), it will perform its function identically in both scenarios. The difference lies in the robustness and source of those inputs.