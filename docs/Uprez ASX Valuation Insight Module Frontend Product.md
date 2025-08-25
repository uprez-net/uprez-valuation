# Uprez ASX Valuation Insight Module: Frontend Product Requirement Document

Status: Not Started

### **1. Introduction & Overview**

This document specifies the frontend requirements for the **Valuation Insight Module**, a new premium feature within the Uprez ASX platform. This module will guide users through a data-gathering workflow to generate an AI-powered, indicative IPO valuation range and price per share.

The primary goal is to create a clean, minimalist, and intuitive user experience that simplifies a complex process, leveraging a conditional workflow based on whether the user has a draft Prospectus. The module will intelligently re-use documents already uploaded during the compliance check to minimize user effort.

**Key User Personas:**

- **Anjali (SME CFO/Finance Head):** Primary user providing the data.
- **Raj (Lead Manager/Corporate Advisor):** Primary consumer of the final report, who will review and add his expert opinion.

**Technical Philosophy:**

- **Single-Page Workflow:** The entire data collection process will occur on a single, scrollable page to maintain context and flow.
- **Conditional Logic:** The UI will dynamically change based on the user's answers, specifically the "Prospectus Yes/No" question.
- **Component-Based UI:** Built using our existing React/Next.js and shadcn/ui component library for consistency.
- **Seamless Backend Integration:** All data submission and processing triggers will be handled via the existing BFF (Next.js API Routes).

---

### **2. User Journey & Core Workflow**

**Entry Point:**

A new Call-to-Action (CTA) button labeled **"Generate Valuation Insight"** will be prominently displayed on the main client dashboard.

**Workflow Logic Gate (The First Interaction):**

1. **Compliance Check Verification:**
    - **Logic:** When the user clicks the "Generate Valuation Insight" CTA, the system first checks if the user has completed the main "IPO Compliance Check" module.
    - **UI (If Compliance Not Complete):** A modal dialog appears.
        - **Header:** "Compliance Check Required"
        - **Body:** "To generate an accurate valuation, our AI needs to analyze the core compliance documents you've already provided. Please complete the IPO Compliance Check first."
        - **CTA:** A single button, **"Go to Compliance Checklist"**, which redirects the user to that module.
    - **UI (If Compliance Is Complete):** The user is navigated to the new /valuation page, and the workflow begins.

**The /valuation Page Workflow:**

This is a single, stateful page that builds itself dynamically.

1. **The Prospectus Question (The Fork in the Road):**
    - The user is presented with a clean, centered modal or initial screen section.
    - **UI:**
        - **Question:** "Do you have a draft or final Prospectus document to analyze?"
        - **Buttons:** Two large, clear buttons: [ Yes, I have a Prospectus ] and [ No, not yet ].
2. **Path A: "Yes, I have a Prospectus"**
    - The initial question section smoothly transitions (e.g., slides up) to reveal the next step.
    - **UI Step 1: Upload Prospectus:**
        - **Header:** "Step 1: Upload Your Prospectus"
        - **Instruction:** "Please upload your most recent draft or final Prospectus. Our AI will analyze it to extract your company narrative, growth strategy, and risk factors."
        - **Component:** A large drag-and-drop file upload component (powered by Vercel Blob/GCP signed URL logic). It will show upload progress.
    - **UI Step 2: Define Peer Group:**
        - **Header:** "Step 2: Identify Your Competitors"
        - **Instruction:** "List the ASX ticker codes of 5-10 publicly listed companies you consider your main competitors. This is crucial for benchmarking your valuation."
        - **Component:** A dynamic tag input field where the user can type tickers (e.g., "XRO"). **AI Enhancement:** As the user types, an autocomplete dropdown appears, pulling from the ASX list of tickers to prevent typos. (not required)
    - **UI Step 3: Generate Report:**
        - **Component:** A final CTA button, [ Generate Valuation Report ].
        - **Action:** Clicking this button triggers the /api/generation/trigger endpoint with all the collected data. The UI then displays a "Processing" state.
3. **Path B: "No, not yet" (The Detailed Workflow)**
    - This path is for users earlier in the IPO process. The page will be a single, scrollable form with distinct sections.
    - **UI Section 1: Core Financials**
        - **Header:** "Step 1: Core Financials & Capital Structure"
        - **Instruction:** "First, let's establish your financial baseline. We have pre-filled what we can from your compliance documents."
        - **Data Point A (Pre-filled):**
            - **Label:** "Audited Financials"
            - **UI:** Shows the filenames of the financial statements already uploaded in the compliance module, with a green checkmark. (e.g., ✔ Audited_FY23.pdf). The user does not need to re-upload.
        - **Data Point B (New Upload):**
            - **Label:** "Current Capitalisation Table"
            - **Instruction:** "Please upload your current Cap Table. This is essential for understanding your pre-IPO share structure."
            - **Component:** A dedicated upload component for this single file.
        - **Data Point C (User Input):**
            - **Label:** "Projected Net Profit After Tax (NPAT) for next financial year"
            - **Component:** A formatted input field for currency (A$).
    - **UI Section 2: Narrative & Growth Strategy**
        - **Header:** "Step 2: Tell Us Your Story"
        - **Instruction:** "In the space below, please summarize your company's vision, the problem you solve, and your key strategies for growth over the next 3 years."
        - **Component A (User Input):** A large, rich-text input area (textarea) for the user to write or paste their narrative.
        - **Component B (Supporting Documents):**
            - **Instruction:** "To support your summary, please upload at least one of the following documents. Our AI will use them to verify and enhance its analysis."
            - **UI:** A multi-file upload component. A message below will state "0 files uploaded. At least 1 file is required." which updates as files are added.
    - **UI Section 3: Risk Factors**
        - **Header:** "Step 3: Identify Key Risks"
        - **Instruction:** "Please briefly describe the top 3-5 risks your business faces."
        - **Component A (User Input):** A dynamic list component where users can add short text descriptions of each risk.
        - **Component B (Supporting Documents):**
            - **Instruction:** "To provide context, please upload any relevant documents like an internal risk register or board papers."
            - **UI:** Another multi-file upload component (optional).
    - **UI Section 4: IPO Structure & Peer Group**
        - **Header:** "Step 4: The IPO Deal & Your Competitors"
        - **Instruction:** "Finally, provide some assumptions about the IPO and your key competitors."
        - **Component A (User Input - Sliders):**
            - "How much capital do you plan to raise?" (Slider from A50M)
                
                ```
                5MtoA5MtoA
                ```
                
            - "What percentage of the company will be sold?" (Slider from 15% to 40%)
        - **Component B (User Input - Tickers):** Same dynamic tag input field for ASX ticker codes as in Path A.
    - **UI Section 5: Generate Report:**
        - Same final CTA button and "Processing" state as in Path A.
4. **The "Processing" State (SLA Screen):**
    - Once the user clicks "Generate," the main content area is replaced with a clean, reassuring screen.
    - **UI:**
        - **Icon:** An animated processing/loading icon.
        - **Header:** "Analyzing Your Business..."
        - **Body:** "Our AI is now processing your documents and market data. This can take several minutes. You can safely navigate away from this page. We will notify you via email and an in-app notification as soon as your Valuation Insight Report is ready."

---

### **3. Frontend Technical Details & Components**

- **State Management:**
    - **Redux Toolkit :** Will be used to manage the multi-step state of the "No Prospectus" workflow, tracking which sections are complete and valid before enabling the final CTA.
- **Key Components to Build:**
    - ValuationEntryGate.tsx: The initial component that checks compliance status and shows the redirect modal if necessary.
    - ProspectusQuestion.tsx: The first interactive step with the "Yes/No" buttons.
    - ValuationWorkflow.tsx: The main parent component that lives on the /valuation page and orchestrates which steps are shown.
    - FileUploadZone.tsx: A reusable component that integrates with the Vercel Blob/GCP logic (/api/uploads/generate-url). It will handle the UI for dragging, dropping, progress bars, and success/error states.
    - PeerGroupInput.tsx: The specialized input for ASX tickers with debounced API calls for the autocomplete feature.
    - ProcessingSLA.tsx: The final screen shown after the user triggers the analysis.
- **API Interactions:**
    - GET /api/compliance/status: Called on page load to determine if the user can start the workflow.
    - POST /api/uploads/generate-url: Called by the FileUploadZone before each file upload.
    - POST /api/valuation/generate: This is the final API call triggered by the "Generate" CTA. It will send a JSON payload containing all the user inputs and the URLs of all uploaded files to the BFF, which then proxies it to the Python backend.

[**Product Requirement Document: The Valuation Insight Report Page**](Uprez%20ASX%20Valuation%20Insight%20Module%20Frontend%20Produc%2025aaa00510bd80a3b75ff3bad5ba20fc/Product%20Requirement%20Document%20The%20Valuation%20Insight%2025aaa00510bd808fbe73c7d729dd11ed.md)