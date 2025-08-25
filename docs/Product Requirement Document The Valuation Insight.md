# Product Requirement Document: The Valuation Insight Report Page

**Objective:** To present the results of the AI-powered valuation analysis in a professional, intuitive, and shareable format. The page will serve both the SME (Anjali) as a strategic planning tool and the Lead Manager (Raj) as a data-driven foundation for their advisory work.

**Access:** Users will access this page from a link in their notification, or via a "View Report" button on their main dashboard once the processing is complete.

---

### **1. Page Structure & Layout**

The page will be a clean, single-view dashboard designed for both on-screen viewing and easy export to PDF. It will be structured into four main horizontal sections.

**Header Section:**

- **Left:** Company Logo and Name.
- **Center:** Main Title: **"Indicative IPO Valuation Report"**.
- **Right:**
    - "Report Generated: 18-Aug-2025"
    - "Status: Final"
    - A primary CTA button: [ ↓ Download PDF ]
    - A secondary CTA button: [ ↻ Re-run Analysis ] (This would take the user back to the data input workflow).

---

### **2. Section 1: The "At-a-Glance" Insight**

**Purpose:** To immediately deliver the most critical, top-line valuation metrics. This is the executive summary.

**UI Layout:** This section will feature 3-4 large "Key Metric Cards" arranged horizontally.

| **Component** | **UI Elements & Data Display** | **Data Points Required (from Step 3)** |
| --- | --- | --- |
| **Card 1: Indicative Pre-Money Valuation** | **Title:** "Indicative Pre-Money Valuation" <br> **Main Display:** A large, bold font showing the range, e.g., **"A

`71.5M−A71.5*M*−*A*`96.7M"** <br> **Subtitle:** "Based on a central valuation of A$84.1M" | indicative_valuation_range.low_end_aud, indicative_valuation_range.central_aud, indicative_valuation_range.high_end_aud |
| **Card 2: Indicative Price Per Share** | **Title:** "Indicative Price Per Share" <br> **Main Display:** A large, bold font showing the price range, e.g., **"A

`1.43−A1.43−*A*`1.94"** <br> **Subtitle:** "Based on [53.3M] total shares post-IPO" | offering_price_range.low_price_aud, offering_price_range.high_price_aud, post_ipo_shares |
| **Card 3: Target Valuation Multiple** | **Title:** "AI-Calculated Target Multiple" <br> **Main Display:** A large, bold font, e.g., **"26.5x P/E"** <br> **Subtitle:** "vs. Peer Median of 22.5x P/E" | target_multiple.value, target_multiple.type, baseline_multiples.pe_ratio_median |
| **Card 4: Capital Raise Potential (Optional)** | **Title:** "Capital Raise at Mid-Point" <br> **Main Display:** A large, bold font, e.g., **"A$20.0M"** <br> **Subtitle:** "Implies a 23.8% stake for new investors" | deal_assumptions.capital_raise, offering_price_range (central price), post_ipo_shares |

---

### **3. Section 2: The Valuation Bridge (Visual Analysis)**

**Purpose:** To visually explain *how* the platform arrived at the target multiple. This is the most important section for building trust and transparency in the AI's analysis.

**UI Layout:** A horizontal "waterfall" or "bridge" chart.

**Component: Valuation Bridge Chart**

- **Chart Title:** "How We Calculated Your Target Multiple"
- **Visual Style:** A professional waterfall chart (using a library like Recharts or Chart.js).
- **Flow of the Chart:**
    1. **Starting Bar (Left):** A solid blue bar labeled **"Peer Median P/E"**, with the value **"22.5x"**.
    2. **Green Bar (Up):** An upward green bar labeled **"Growth & Narrative Premium"**, with the value **"+4.2x"**. *Hovering over this bar reveals a tooltip with the justification from the AI model (e.g., "Strong growth story backed by historical data...").*
    3. **Red Bar (Down):** A downward red bar labeled **"Risk Discount"**, with the value **"-4.0x"**. *Hovering reveals the key risks identified (e.g., "Discount applied due to single customer dependency...").*
    4. **Green Bar (Up):** Another upward green bar labeled **"Market & Sector Tailwind"**, with the value **"+3.8x"**. *Hovering reveals the drivers (e.g., "Positive market sentiment and high hype score for AI sector...").*
    5. **Final Bar (Right):** A final solid blue bar labeled **"Your Target P/E"**, ending at the value **"26.5x"**.

**Data Points Required:**

- baseline_multiples.pe_ratio_median
- The raw values of the adjustments calculated from growth_narrative_score, risk_weighting_factor, and market_adjustment_factor from Step 2. The backend will need to provide not just the final multiple, but the value of each adjustment step.

---

### **4. Section 3: Interactive Scenario Analysis**

**Purpose:** To turn the static report into a dynamic tool, allowing the user to model different IPO scenarios.

**UI Layout:** A clean table with input sliders above it.

**Component: IPO Scenario Modeler**

- **Title:** "Model Your IPO Scenarios"
- **Interactive Inputs (Sliders):**
    - **Slider 1:** "Target Capital Raise:" (User can drag from A50M).
        
        ```
        5MtoA5MtoA
        ```
        
    - **Slider 2:** "Issue Price Per Share:" (User can drag within the AI-generated "sweet spot" range, e.g., $1.43 to $1.94).
- **Output Table (updates in real-time as sliders are moved):**
    - A table with the following rows:

| **Metric** | **Value** | **Explanation** |
| --- | --- | --- |
| **Issue Price** | A$1.75 | The price per share you are offering. |
| **Capital to be Raised** | A$20.0M | The total funds you are raising in the IPO. |
| **New Shares to be Issued** | 11,428,571 | The number of new shares you need to issue to raise the capital at this price. |
| **Company Stake Sold (Dilution)** | 21.4% | The percentage of the company new investors will own post-IPO. |
| **Implied Post-Money Valuation** | A$93.3M | The total value of the company immediately after the IPO. |

**Data Points Required:**

- pre_ipo_shares (from Step 1).
- The offering_price_range (to set the bounds of the slider).

### **5. Section 4: AI Analysis Deep Dive (The Appendix)**

**Purpose:** To provide full transparency and allow expert users (like Raj) to drill down into the AI's specific findings and evidence.

**UI Layout:** A tabbed interface for a clean layout.

**Component: Tabbed Evidence Panel**

- **Tab 1: Growth & Narrative Analysis**
    - **AI Summary:** Displays the AI's summary of the company's vision and strategy.
    - **Evidence Table:** A table listing each key growth driver the AI identified.
        
        | Growth Driver                             | Source Document(s)                                    | AI Confidence |
        
        | Expansion into the UK Market | Investor_Deck.pptx, Board_Minutes.pdf | 95% |
        
        | Launch of Predictive Analytics Module | Business_Plan.pdf                         | 92% |
        
- **Tab 2: Risk Factor Analysis**
    - **AI Summary:** A brief summary of the overall risk profile.
    - **Evidence Table:** A table listing each key risk identified.
        
        | Risk Identified                             | Mitigation Status| Source Document(s)                               | Severity |
        
        | Single Customer Dependency| Mitigated             | Risk_Register.xlsx, Board_Minutes.pdf | High -> Medium |
        
        | Key Person Risk (CTO)                | Partially Mitigated| Board_Minutes.pdf                                 | High -> High |
        
- **Tab 3: Comparable Company Data**
    - **Data Table:** A detailed table showing the full list of peer companies and their individual valuation metrics (P/E, EV/EBITDA, etc.) that were used to calculate the median. This allows the Lead Manager to validate the peer set.

**Data Points Required:**

- All the detailed, evidence-linked JSON outputs from Step 2.