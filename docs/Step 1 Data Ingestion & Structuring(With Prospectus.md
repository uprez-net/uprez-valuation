# Step 1: Data Ingestion & Structuring(With Prospectus)

### **Valuation Engine - Step 1: Data Ingestion & Structuring**

**Objective:** To programmatically gather, extract, and structure all the raw data points required for the three analytical dimensions. This step transforms unstructured documents and external data feeds into a clean, machine-readable format ready for the synthesis engine.

---

### **Source 1: User-Uploaded Documents (Internal Company Data)**

This data is provided by the user (the SME or their advisor) and forms the foundation of the company-centric analysis. The platform's AI will use a combination of OCR, table extraction, and Natural Language Processing (NLP) to process these documents.

### **A. The Prospectus (or Draft Prospectus)**

This is the single most important document. It's a goldmine of both quantitative and qualitative data.

| Data Point | Source & How to Get It | Importance | Processing Method | Stored Output (Example) |
| --- | --- | --- | --- | --- |
| **Company Narrative** | **Source:** "Chairman's Letter", "CEO's Letter", "Investment Overview" sections. <br> **How:** OCR the document, then use NLP to isolate and analyze these specific sections. | **High** | **NLP - Sentiment & Thematic Analysis:** <br> 1. Run sentiment analysis on the text to gauge the tone (positive, negative, neutral). <br> 2. Use topic modeling or keyword extraction to identify the core themes of the company's story (e.g., "disruption", "market leadership", "innovation"). | {"narrative_sentiment": 0.85, "narrative_themes": ["innovation", "market expansion", "experienced team"]} |
| **Growth Strategy** | **Source:** "Business Model", "Growth Strategy", "Future Plans" sections. <br> **How:** OCR + NLP to identify sections discussing future growth. | **Critical** | **NLP - Named Entity Recognition & Keyword Extraction:** <br> 1. Identify specific growth drivers (e.g., new products, geographic expansion, key partnerships). <br> 2. Quantify the drivers where possible (e.g., "expand into the UK market," "launch Product X in Q3"). | {"growth_drivers": [{"type": "geography", "detail": "UK market entry"}, {"type": "product", "detail": "Launch Product X"}, {"type": "strategy", "detail": "Key partnership with BigCorp"}]} |
| **Risk Factors** | **Source:** The dedicated "Risk Factors" section. <br> **How:** OCR + NLP to extract the full list of disclosed risks. | **Critical** | **NLP - Classification & Severity Scoring:** <br> 1. Classify each risk into categories (e.g., Market, Operational, Regulatory, IP, Personnel). <br> 2. Use sentiment analysis and keyword matching (e.g., "highly dependent", "significant risk", "no assurance") to assign a severity score. | {"risks": [{"category": "Operational", "description": "Dependence on a single key supplier...", "severity_score": 0.9}, {"category": "Market", "description": "Increased competition from new entrants...", "severity_score": 0.6}]} |
| **Capital Structure (Post-IPO)** | **Source:** The "Capital Structure" table within the Prospectus. <br> **How:** OCR + Table Extraction. Isolate the table and extract the row for "Total Shares on Issue" post-completion of the offer. | **Critical** | **Direct Value Extraction:** This is a hard number, not an interpretation. The system must find this specific value. It will be used as a denominator in the final price-per-share calculation. | {"post_ipo_shares": 50000000} |
| **Use of Funds** | **Source:** The "Use of Funds" table. <br> **How:** OCR + Table Extraction. | **Medium** | **Structured Data Extraction:** Extract the line items and the corresponding dollar amounts or percentages allocated to each. This helps verify the Working Capital statement and provides context on the company's priorities. | {"use_of_funds": [{"item": "R&D", "amount": 10000000}, {"item": "Marketing", "amount": 8000000}, {"item": "Costs of the Offer", "amount": 2000000}]} |

---

### **B. Financial Model / Forecasts**

This is typically an Excel or CSV file. It's the most sensitive and important quantitative input.

| Data Point | Source & How to Get It | Importance | Processing Method | Stored Output (Example) |
| --- | --- | --- | --- | --- |
| **Projected Net Income / NPAT** | **Source:** User upload of a financial model (Excel/CSV). <br> **How:** The platform will scan the uploaded spreadsheet for specific keywords like "Net Profit After Tax", "NPAT", "Net Income" for the next full financial year (FY+1). **Fallback:** If the AI cannot confidently identify the number, the UI must prompt the user to manually enter or confirm the value. | **Critical** | **Keyword-Based Cell Extraction & User Verification:** <br> 1. Parse the spreadsheet to find the target cell. <br> 2. Present the extracted value to the user for confirmation ("We've identified your FY+1 NPAT as A$3,000,000. Is this correct?"). | {"projected_npat": 3000000} |

---

### **C. Audited Financial Statements (for last 2 years)**

These are PDF documents that validate the company's historical performance and asset base.

| Data Point | Source & How to Get It | Importance | Processing Method | Stored Output (Example) |
| --- | --- | --- | --- | --- |
| **Historical Performance** | **Source:** "Statement of Profit or Loss and Other Comprehensive Income". <br> **How:** OCR + Table Extraction. | **High** | **Time-Series Data Extraction:** Extract Revenue, EBITDA, and Net Profit for FY-1 and FY-2 to establish a historical growth trajectory. | {"historical_performance": [{"year": "FY-1", "revenue": 15M, "ebitda": 4M, "npat": 2M}, {"year": "FY-2", "revenue": 10M, "ebitda": 2.5M, "npat": 1M}]} |
| **Net Tangible Assets (NTA)** | **Source:** "Statement of Financial Position" (Balance Sheet). <br> **How:** OCR + Table Extraction. | **High** | **Formulaic Calculation:** Extract Total Assets, Intangible Assets, and Total Liabilities. Calculate NTA = (Total Assets - Intangible Assets - Total Liabilities). | {"nta_value": 4500000} |
| **Auditor's Opinion** | **Source:** The "Independent Auditor's Report" section. <br> **How:** OCR + NLP. Scan the report for keywords like "unqualified opinion", "qualified opinion", "disclaimer of opinion", or "emphasis of matter". | **Critical** | **Keyword-Based Classification:** Classify the audit report's outcome. A "qualified" opinion is a major red flag that must be highlighted. | {"audit_opinion_fy1": "Unqualified"} |

---

### **Source 2: External APIs (Market & Economic Data)**

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