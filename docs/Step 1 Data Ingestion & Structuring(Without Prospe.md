# Step 1: Data Ingestion & Structuring(Without Prospectus)

### **Valuation Engine - Step 1 (Redefined): Pre-Prospectus Data Ingestion & Structuring**

**Objective:** To programmatically gather, extract, and structure all raw data points required for the valuation analysis, assuming no Prospectus is available. The platform will rely on a combination of core financial documents, strategic internal documents, and structured user inputs.

---

### **Source 1: User-Uploaded Documents (Internal Company Data)**

This is now the primary source for all company-specific information. The user (Anjali, the CFO) will be guided to upload a specific set of documents that exist *before* a Prospectus is drafted.

### **A. Core Financial Documents (The "Hard Numbers")**

| **Data Point** | **Source & How to Get It** | **Importance** | **Processing Method** | **Stored Output (Example)** |
| --- | --- | --- | --- | --- |
| **Historical Performance** | **Source:** **Audited Financial Statements** (PDFs for the last 2-3 years). <br> **How:** User uploads the PDFs. The platform uses OCR and table extraction to parse the "Statement of Profit or Loss." | **High** | **Time-Series Data Extraction:** Extract line items for Revenue, EBITDA, and Net Profit to establish a historical growth trajectory. The AI must be trained to recognize various accounting formats. | {"historical_performance": [{"year": "FY-1", "revenue": 15M, "ebitda": 4M, "npat": 2M}, ...] |
| **Net Tangible Assets (NTA)** | **Source:** **Audited Financial Statements** (Balance Sheet section). <br> **How:** OCR and table extraction on the "Statement of Financial Position." | **High** | **Formulaic Calculation:** Extract Total Assets, Intangible Assets, and Total Liabilities. Calculate NTA = (Total Assets - Intangible Assets - Total Liabilities). | {"nta_value": 4500000} |
| **Auditor's Opinion** | **Source:** **Audited Financial Statements**. <br> **How:** OCR and NLP on the "Independent Auditor's Report" section of each financial statement. | **Critical** | **Keyword-Based Classification:** Scan for keywords like "unqualified opinion", "qualified opinion", "emphasis of matter". A qualified opinion is a major red flag. | {"audit_opinion_fy1": "Unqualified"} |
| **Capital Structure (Current)** | **Source:** The company's current **Capitalisation Table ("Cap Table")**. This is a crucial document, usually maintained in a spreadsheet. <br> **How:** User uploads the Cap Table (Excel/CSV). The platform parses it to get the current share count. | **High** | **Structured Data Extraction:** Parse the spreadsheet to find the total number of ordinary shares, options, and performance rights currently on issue. This is the baseline before modeling the IPO. | {"pre_ipo_shares": 40000000} |

### **B. Strategic & Governance Documents (The "Qualitative Story")**

### **1. Company Narrative & Growth Strategy**

This is the most important qualitative input. It defines the "story" and justifies the future growth that investors are buying into.

**UI/UX Implementation:**

Instead of a simple "Upload Document" button, the UI will be structured as a guided prompt:

> Tell Us Your Story: Company Vision & Growth Strategy
> 
> 
> To help us understand your company's potential, please provide the documents that best describe your vision, market opportunity, and plans for future growth. Our AI will analyze these to quantify the strength of your growth story.
> 
> **You can upload one or more of the following:**
> 
> - Investor Presentation / Pitch Deck (PPTX, PDF)
> - Information Memorandum (IM) from a previous capital raise
> - Formal Business Plan
> - Board Strategy Documents or Offsite Presentations
> 
> *Drag and drop files here or click to browse.*
> 
> **[Upload Area]**
> 

**How the AI Processes the Uploaded Documents:**

The backend process is now more sophisticated because it cannot assume a fixed document structure.

1. **Document Type Identification (Initial Classification):**
    - The AI first analyzes the structure and keywords of each uploaded document to classify it (e.g., "This looks like a PowerPoint presentation," "This document contains financial tables and legal disclaimers, likely an IM"). This helps in prioritizing which sections to analyze.
2. **Thematic Section Search (The Core Logic):**
    - The AI doesn't look for a "Growth Strategy" chapter heading. Instead, it performs a **semantic search** across the entire content of all uploaded documents for key themes. It's trained to find sections that answer questions like:
        - "What is the company's mission or vision?"
        - "What problem does the company solve?"
        - "How large is the total addressable market (TAM)?"
        - "What is the go-to-market strategy?"
        - "What are the key growth drivers for the next 3-5 years?" (e.g., new products, geographic expansion, M&A).
        - "Who are the key members of the management team?"
3. **Data Extraction and Structuring:**
    - Once these thematic sections are identified, the AI extracts the relevant text and data points.
    - **Sentiment Analysis:** It analyzes the tone of the language used (positive, confident vs. vague, uncertain).
    - **Keyword & Entity Recognition:** It pulls out specific, quantifiable claims (e.g., "projected 50% CAGR," "expand to the UK market," "launch Product X").
    - **Source Attribution:** The stored output will now include the source document for each piece of evidence. This is crucial for the Lead Manager (Raj) to verify the AI's findings.

**Example Stored Output (More Detailed):**

codeJSON

```
{
  "narrative_summary": {
    "vision_statement": "To become the leading provider of AI-driven logistics in Australia.",
    "market_problem": "Inefficient last-mile delivery...",
    "solution": "Our proprietary routing software..."
  },
  "growth_drivers": [
    {
      "type": "geography",
      "detail": "Expansion into the New Zealand market in FY+2.",
      "source_document": "Investor_Deck_Q3.pptx",
      "confidence_score": 0.95
    },
    {
      "type": "product",
      "detail": "Launch of a new predictive analytics module.",
      "source_document": "Internal_Business_Plan.pdf",
      "confidence_score": 0.92
    }
  ],
  "management_highlights": ["CEO: John Doe (ex-Uber)", "CTO: Jane Smith (PhD in AI)"]
}
```

---

### **2. Risk Factors (Initial Assessment)**

This is equally critical for grounding the valuation in reality.

**UI/UX Implementation:**

The prompt will be similar, asking for documents that outline the company's challenges.

> Understanding the Risks: Challenges & Mitigations
> 
> 
> A realistic valuation must consider the potential risks to your business. Please upload any documents where your company has formally identified and discussed potential risks, threats, or challenges.
> 
> **Good examples include:**
> 
> - Internal Risk Register or Matrix
> - Due Diligence Questionnaires (DDQs)
> - Board Meeting Minutes or Papers discussing risks
> - SWOT Analysis from a business plan
> 
> *Drag and drop files here or click to browse.*
> 
> **[Upload Area]**
> 

**How the AI Processes the Uploaded Documents:**

1. **Targeted Semantic Search:** The AI scans all uploaded documents specifically for sections or sentences related to risk. It searches for:
    - Headings like "Risk Factors," "Key Risks," "Threats," "Challenges."
    - Keywords and phrases like "dependent on," "risk of," "failure to," "competition," "regulatory hurdles," "litigation," "key person risk."
2. **Risk Extraction and Classification:**
    - When a potential risk is identified, the AI extracts the full statement.
    - It then uses a **text classification model** to categorize the risk (e.g., Market, Operational, Financial, Legal/Regulatory, Personnel).
3. **Severity Scoring and Mitigation Analysis:**
    - The AI assigns a preliminary **severity score** based on the language used (e.g., "significant risk," "materially and adversely affect," "no assurance" result in higher scores).
    - Crucially, it also searches the surrounding text for any stated **mitigation strategies**. The presence of a clear, credible mitigation plan can reduce the final risk weighting.

**Example Stored Output (More Detailed):**

codeJSON

```
{
  "risks": [
    {
      "category": "Operational",
      "description": "The company is highly dependent on a single key supplier for its primary raw material.",
      "severity_score": 0.9,
      "mitigation_strategy": "Management is in the process of qualifying a second supplier, expected completion in Q4.",
      "source_document": "Risk_Register_FY24.xlsx",
      "confidence_score": 0.98
    },
    {
      "category": "Personnel",
      "description": "Loss of our CTO, Dr. Jane Smith, could significantly impact our R&D roadmap.",
      "severity_score": 0.8,
      "mitigation_strategy": "Key person insurance is in place; knowledge transfer protocols are being documented.",
      "source_document": "Board_Minutes_July.pdf",
      "confidence_score": 0.91
    }
  ]
}
```

### **Benefits of This Approach for the Developer**

- **Flexibility & Robustness:** The system doesn't break if the user doesn't have a perfectly formatted "Business Plan." It can ingest and analyze a variety of real-world documents.
- **Enhanced AI Challenge:** This moves the AI from simple extraction to a more complex task of **thematic understanding and cross-document synthesis**. This is a powerful differentiator.
- **Superior User Experience:** The user is not forced to create a new document for the platform. They can use the strategic documents they already have, saving them significant time and effort.
- **Traceability:** By including source_document and confidence_score in the stored output, the system provides a clear audit trail. The Lead Manager (Raj) can easily click a link to see the exact sentence in the exact document where the AI found its information, allowing for quick human verification.

---

### **Source 2: Structured User Inputs (Filling the Gaps)**

Because there is no Prospectus, the platform must now actively prompt the user for key information that cannot be reliably extracted from the documents above. This is a critical UI/UX consideration.

| **Data Point** | **Source & How to Get It** | **Importance** | **Processing Method** | **Stored Output (Example)** |
| --- | --- | --- | --- | --- |
| **Projected Net Income / NPAT** | **Source:** **Direct User Input**. <br> **How:** The platform will have a dedicated, mandatory input field in the valuation module: *"Please enter your company's projected Net Profit After Tax (NPAT) for the next full financial year (e.g., A$3,000,000)."* The user must provide this number. | **Critical** | **Direct Value Ingestion:** The value is taken directly from the user input field. The platform can cross-reference it with the uploaded Financial Model if one is provided, and flag any major discrepancies. | {"projected_npat": 3000000} |
| **Comparable Peer Group** | **Source:** **Direct User Input (with AI assistance)**. <br> **How:** The user is presented with an input field to list the ASX ticker codes of their main competitors. **AI Enhancement:** The platform could pre-populate this list with suggestions based on an analysis of the company's website or business plan. | **Critical** | **User Input Capture:** The list of ticker codes is captured and used to query the external Market Data API. | {"peer_group_tickers": ["XRO", "NXL", "IRE"]} |
| **IPO Deal Structure Assumptions** | **Source:** **Direct User Input**. <br> **How:** The platform will ask the user for key assumptions about the IPO itself, as these are needed to calculate the final price per share. The UI will have sliders or input fields for: <br> - "How much capital do you plan to raise? (e.g., A$20M)" <br> - "What percentage of the company will be sold in the IPO? (e.g., 25%)" | **High** | **Formulaic Calculation:** The platform uses these inputs to calculate the **Post-IPO Shares on Issue**, which is a crucial denominator. <br> Post-IPO Shares = Pre-IPO Shares / (1 - % Sold) | {"deal_assumptions": {"capital_raise": 20000000, "sell_down_percent": 0.25}, "post_ipo_shares": 53333333} |

---

### **Source 3: External APIs (Market & Economic Data)**

This data provides the external, real-time context for the valuation. The platform will need subscriptions to one or more financial data APIs.

### **A. ASX Market Data API (e.g., Refinitiv, S&P Capital IQ)**

This is used to get data on the comparable peer companies.

| Data Point | Source & How to Get It | Importance | Processing Method | Stored Output (Example) |
| --- | --- | --- | --- | --- |
| **Peer Valuation Multiples** | **Source:** Financial data API endpoint for equities. <br> **How:** For each ASX ticker code in the user-defined peer group, make an API call to fetch their current Market Cap, Enterprise Value, trailing 12-month (TTM) Revenue, EBITDA, and Net Profit. | **Critical** | **Calculation & Aggregation:** <br> 1. For each peer, calculate P/E, EV/EBITDA, and EV/Sales ratios. <br> 2. Discard any extreme outliers. <br> 3. Calculate the **median** of each ratio across the peer group. | {"peer_group_medians": {"pe_ratio": 25.5, "ev_ebitda_ratio": 12.1, "ev_sales_ratio": 4.5}} |

---

### **B. Market Sentiment & Economic Data APIs**

| Data Point | Source & How to Get It | Importance | Processing Method | Stored Output (Example) |
| --- | --- | --- | --- | --- |
| **Market Fear Gauge** | **Source:** Market data API that provides index data. <br> **How:** Fetch the current value and 30-day trend for the **S&P/ASX 200 VIX (XVI) index**. | **High** | **Trend Analysis:** Compare the current value to the 30-day average to determine if market fear is rising, falling, or stable. | {"vix_data": {"current_value": 12.5, "trend": "falling"}} |
| **Macroeconomic Indicators** | **Source:** APIs from the **RBA** and **ABS**, or a consolidated provider. <br> **How:** Fetch the latest published figures for the Official Cash Rate, CPI (Inflation), quarterly GDP growth, and the unemployment rate. | **Medium** | **Direct Value Ingestion:** These are hard numbers that provide context. They are stored directly. | {"macro_data": {"cash_rate": 4.35, "cpi_yoy": 3.6, "gdp_qoq": 0.2, "unemployment_rate": 4.1}} |
| **Sector Hype/Momentum** | **Source:** Financial News API (e.g., covering AFR, The Australian) and Broker Research databases. <br> **How:** Run NLP queries to track the volume of positive-sentiment mentions of the company's sector keywords (e.g., "AI", "Lithium", "Cybersecurity") over the past 3-6 months. | **Medium** | **Frequency & Trend Analysis:** Analyze the velocity of keyword mentions. A sharp increase in positive coverage indicates a "hot" sector. This is translated into a score. | {"sector_momentum": {"sector": "AI", "hype_score": 0.85, "trend": "rising"}} |
| **Precedent M&A Transactions** | **Source:** M&A databases or by scanning ASX announcements via a News API. <br> **How:** Search for takeover and M&A announcements within the company's sector over the last 24 months. Extract the deal value and the target's financials to calculate the transaction multiple. | **Medium** | **Data Extraction & Calculation:** The goal is to build a small, relevant dataset of recent M&A multiples to compare against the public market multiples. | {"precedent_transactions": [{"target": "HealthTechAus", "acquirer": "BigCorp", "deal_value": 100M, "ev_ebitda_multiple": 15.0}]} |