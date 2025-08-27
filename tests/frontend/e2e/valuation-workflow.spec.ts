/**
 * End-to-end tests for the complete IPO valuation workflow.
 * 
 * Tests the full user journey from login through valuation creation,
 * monitoring progress, and viewing results.
 */

import { test, expect, Page } from '@playwright/test';

// Test data
const testCompany = {
  name: 'TechCorp Australia Pty Ltd',
  abn: '12345678901',
  industry: 'Technology',
  description: 'An innovative AI-powered fintech platform that provides SME lending solutions with advanced risk assessment and automated decision-making capabilities.',
  ipoDate: '2024-12-31',
  sharesIssued: '50,000,000',
  minPrice: '2.50',
  maxPrice: '4.00'
};

const testUser = {
  email: 'test.user@techcorp.com.au',
  password: 'SecureTestPassword123!',
  firstName: 'John',
  lastName: 'Smith',
  company: 'TechCorp Australia'
};

// Helper functions
async function loginUser(page: Page) {
  await page.goto('/login');
  
  await page.fill('[data-testid="email-input"]', testUser.email);
  await page.fill('[data-testid="password-input"]', testUser.password);
  await page.click('[data-testid="login-button"]');
  
  // Wait for dashboard to load
  await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
  await expect(page).toHaveURL('/dashboard');
}

async function fillValuationForm(page: Page) {
  // Company Information
  await page.fill('[data-testid="company-name-input"]', testCompany.name);
  await page.fill('[data-testid="abn-input"]', testCompany.abn);
  await page.selectOption('[data-testid="industry-select"]', testCompany.industry);
  await page.fill('[data-testid="description-textarea"]', testCompany.description);
  
  // IPO Details
  await page.fill('[data-testid="ipo-date-input"]', testCompany.ipoDate);
  await page.fill('[data-testid="shares-input"]', testCompany.sharesIssued);
  await page.fill('[data-testid="min-price-input"]', testCompany.minPrice);
  await page.fill('[data-testid="max-price-input"]', testCompany.maxPrice);
  
  // Analysis Options (use defaults for comprehensive analysis)
  await expect(page.locator('[data-testid="analysis-type-select"]')).toHaveValue('comprehensive');
  await expect(page.locator('[data-testid="peer-analysis-checkbox"]')).toBeChecked();
  await expect(page.locator('[data-testid="risk-level-select"]')).toHaveValue('detailed');
}

test.describe('IPO Valuation Workflow', () => {
  test.beforeEach(async ({ page }) => {
    // Set up test data
    await page.addInitScript(() => {
      // Mock API responses for testing
      window.__TEST_MODE__ = true;
    });
  });

  test('complete valuation workflow from creation to results', async ({ page }) => {
    // Step 1: Login
    await loginUser(page);
    
    // Verify dashboard shows correct user info
    await expect(page.locator('[data-testid="user-name"]')).toContainText(`${testUser.firstName} ${testUser.lastName}`);
    await expect(page.locator('[data-testid="company-name"]')).toContainText(testUser.company);
    
    // Step 2: Navigate to create new valuation
    await page.click('[data-testid="create-valuation-button"]');
    await expect(page).toHaveURL('/valuations/create');
    await expect(page.locator('h1')).toContainText('Create New IPO Valuation');
    
    // Step 3: Fill out valuation form
    await fillValuationForm(page);
    
    // Step 4: Submit valuation request
    await page.click('[data-testid="submit-valuation-button"]');
    
    // Step 5: Verify submission success
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="job-id"]')).toContainText(/val_[a-z0-9]+/);
    
    // Extract job ID for tracking
    const jobIdElement = page.locator('[data-testid="job-id"]');
    const jobIdText = await jobIdElement.textContent();
    const jobId = jobIdText?.match(/val_[a-z0-9]+/)?.[0];
    expect(jobId).toBeTruthy();
    
    // Step 6: Navigate to valuation status page
    await page.click('[data-testid="view-status-button"]');
    await expect(page).toHaveURL(`/valuations/${jobId}/status`);
    
    // Step 7: Monitor valuation progress
    await expect(page.locator('[data-testid="valuation-status"]')).toContainText(/queued|processing/);
    await expect(page.locator('[data-testid="progress-bar"]')).toBeVisible();
    await expect(page.locator('[data-testid="estimated-completion"]')).toBeVisible();
    
    // Wait for processing stages to appear
    await expect(page.locator('[data-testid="processing-stages"]')).toBeVisible();
    
    // Verify processing stages are displayed
    const stages = [
      'document_processing',
      'market_data_collection',
      'peer_analysis',
      'financial_modeling',
      'risk_assessment',
      'valuation_calculation',
      'report_generation'
    ];
    
    for (const stage of stages) {
      await expect(page.locator(`[data-testid="stage-${stage}"]`)).toBeVisible();
    }
    
    // Step 8: Wait for completion (in test environment, this should be quick)
    await page.waitForFunction(() => {
      const statusElement = document.querySelector('[data-testid="valuation-status"]');
      return statusElement?.textContent?.includes('completed');
    }, { timeout: 60000 });
    
    await expect(page.locator('[data-testid="valuation-status"]')).toContainText('completed');
    await expect(page.locator('[data-testid="progress-bar"]')).toHaveAttribute('aria-valuenow', '100');
    
    // Step 9: View results
    await page.click('[data-testid="view-results-button"]');
    await expect(page).toHaveURL(`/valuations/${jobId}/results`);
    
    // Step 10: Verify results page content
    await expect(page.locator('h1')).toContainText('Valuation Results');
    await expect(page.locator('[data-testid="company-name"]')).toContainText(testCompany.name);
    
    // Verify valuation summary
    await expect(page.locator('[data-testid="target-price"]')).toBeVisible();
    await expect(page.locator('[data-testid="price-range"]')).toBeVisible();
    await expect(page.locator('[data-testid="market-cap"]')).toBeVisible();
    await expect(page.locator('[data-testid="confidence-score"]')).toBeVisible();
    
    // Verify analysis sections
    await expect(page.locator('[data-testid="dcf-analysis"]')).toBeVisible();
    await expect(page.locator('[data-testid="peer-analysis"]')).toBeVisible();
    await expect(page.locator('[data-testid="risk-assessment"]')).toBeVisible();
    await expect(page.locator('[data-testid="scenario-analysis"]')).toBeVisible();
    
    // Step 11: Download PDF report
    const downloadPromise = page.waitForEvent('download');
    await page.click('[data-testid="download-report-button"]');
    const download = await downloadPromise;
    
    expect(download.suggestedFilename()).toMatch(/.*_IPO_Valuation_Report\.pdf$/);
    
    // Step 12: Navigate back to dashboard
    await page.click('[data-testid="back-to-dashboard"]');
    await expect(page).toHaveURL('/dashboard');
    
    // Verify the valuation appears in the recent valuations list
    await expect(page.locator('[data-testid="recent-valuations"]')).toBeVisible();
    await expect(page.locator(`[data-testid="valuation-${jobId}"]`)).toBeVisible();
    await expect(page.locator(`[data-testid="valuation-${jobId}"] [data-testid="status"]`)).toContainText('completed');
  });

  test('handles valuation form validation errors', async ({ page }) => {
    await loginUser(page);
    
    // Navigate to create valuation
    await page.click('[data-testid="create-valuation-button"]');
    
    // Try to submit empty form
    await page.click('[data-testid="submit-valuation-button"]');
    
    // Verify validation errors are displayed
    await expect(page.locator('[data-testid="error-company-name"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-industry"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-ipo-date"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-shares"]')).toBeVisible();
    
    // Fix errors one by one and verify they disappear
    await page.fill('[data-testid="company-name-input"]', testCompany.name);
    await expect(page.locator('[data-testid="error-company-name"]')).not.toBeVisible();
    
    await page.selectOption('[data-testid="industry-select"]', testCompany.industry);
    await expect(page.locator('[data-testid="error-industry"]')).not.toBeVisible();
    
    // Test ABN validation
    await page.fill('[data-testid="abn-input"]', '123'); // Invalid ABN
    await page.blur('[data-testid="abn-input"]');
    await expect(page.locator('[data-testid="error-abn"]')).toContainText('Invalid ABN format');
    
    await page.fill('[data-testid="abn-input"]', testCompany.abn); // Valid ABN
    await expect(page.locator('[data-testid="error-abn"]')).not.toBeVisible();
    
    // Test price range validation
    await page.fill('[data-testid="min-price-input"]', '5.00');
    await page.fill('[data-testid="max-price-input"]', '3.00'); // Max less than min
    await page.blur('[data-testid="max-price-input"]');
    await expect(page.locator('[data-testid="error-price-range"]')).toBeVisible();
    
    await page.fill('[data-testid="max-price-input"]', '6.00'); // Fix price range
    await expect(page.locator('[data-testid="error-price-range"]')).not.toBeVisible();
  });

  test('handles document upload workflow', async ({ page }) => {
    await loginUser(page);
    
    // Navigate to create valuation
    await page.click('[data-testid="create-valuation-button"]');
    
    // Fill basic form data
    await page.fill('[data-testid="company-name-input"]', testCompany.name);
    await page.selectOption('[data-testid="industry-select"]', testCompany.industry);
    await page.fill('[data-testid="ipo-date-input"]', testCompany.ipoDate);
    await page.fill('[data-testid="shares-input"]', testCompany.sharesIssued);
    
    // Navigate to document upload section
    await page.click('[data-testid="documents-tab"]');
    
    // Upload prospectus
    const prospectusFile = 'test-files/sample-prospectus.pdf';
    await page.setInputFiles('[data-testid="prospectus-upload"]', prospectusFile);
    
    // Wait for upload completion
    await expect(page.locator('[data-testid="prospectus-upload-success"]')).toBeVisible();
    await expect(page.locator('[data-testid="prospectus-status"]')).toContainText('processing');
    
    // Upload financial statements
    const financialsFile = 'test-files/financial-statements.xlsx';
    await page.setInputFiles('[data-testid="financials-upload"]', financialsFile);
    
    await expect(page.locator('[data-testid="financials-upload-success"]')).toBeVisible();
    
    // Verify document processing status
    await page.waitForFunction(() => {
      const prospectusStatus = document.querySelector('[data-testid="prospectus-status"]');
      const financialsStatus = document.querySelector('[data-testid="financials-status"]');
      return prospectusStatus?.textContent?.includes('completed') && 
             financialsStatus?.textContent?.includes('completed');
    }, { timeout: 30000 });
    
    // Return to form and submit
    await page.click('[data-testid="form-tab"]');
    await page.click('[data-testid="submit-valuation-button"]');
    
    // Verify submission includes document references
    await expect(page.locator('[data-testid="success-message"]')).toContainText('Documents uploaded successfully');
  });

  test('handles peer analysis selection and customization', async ({ page }) => {
    await loginUser(page);
    
    await page.click('[data-testid="create-valuation-button"]');
    
    // Fill basic form
    await fillValuationForm(page);
    
    // Navigate to analysis options
    await page.click('[data-testid="analysis-options-tab"]');
    
    // Customize peer analysis
    await page.check('[data-testid="peer-analysis-checkbox"]');
    await page.click('[data-testid="customize-peers-button"]');
    
    // Verify peer customization modal opens
    await expect(page.locator('[data-testid="peer-customization-modal"]')).toBeVisible();
    
    // Set market cap range
    await page.fill('[data-testid="min-market-cap"]', '100000000'); // 100M
    await page.fill('[data-testid="max-market-cap"]', '5000000000'); // 5B
    
    // Exclude specific companies
    await page.fill('[data-testid="exclude-tickers"]', 'XYZ,ABC');
    
    // Set maximum number of peers
    await page.selectOption('[data-testid="max-peers-select"]', '15');
    
    // Apply customizations
    await page.click('[data-testid="apply-peer-settings"]');
    await expect(page.locator('[data-testid="peer-customization-modal"]')).not.toBeVisible();
    
    // Verify settings are reflected in the form
    await expect(page.locator('[data-testid="peer-settings-summary"]')).toContainText('Custom peer group');
    
    // Submit valuation
    await page.click('[data-testid="submit-valuation-button"]');
    
    // Verify peer analysis is included in processing
    const jobIdElement = page.locator('[data-testid="job-id"]');
    const jobIdText = await jobIdElement.textContent();
    const jobId = jobIdText?.match(/val_[a-z0-9]+/)?.[0];
    
    await page.click('[data-testid="view-status-button"]');
    await expect(page.locator('[data-testid="stage-peer_analysis"]')).toBeVisible();
  });

  test('handles network errors and retry functionality', async ({ page }) => {
    await loginUser(page);
    
    // Mock network failure
    await page.route('/api/v1/valuations', route => {
      route.abort();
    });
    
    await page.click('[data-testid="create-valuation-button"]');
    await fillValuationForm(page);
    await page.click('[data-testid="submit-valuation-button"]');
    
    // Verify error message is displayed
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Network error');
    await expect(page.locator('[data-testid="retry-button"]')).toBeVisible();
    
    // Remove the route mock to simulate network recovery
    await page.unroute('/api/v1/valuations');
    
    // Retry the request
    await page.click('[data-testid="retry-button"]');
    
    // Verify successful submission after retry
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
  });

  test('saves and restores form data from local storage', async ({ page }) => {
    await loginUser(page);
    
    await page.click('[data-testid="create-valuation-button"]');
    
    // Fill partial form
    await page.fill('[data-testid="company-name-input"]', 'Draft Company');
    await page.fill('[data-testid="abn-input"]', '98765432109');
    await page.selectOption('[data-testid="industry-select"]', 'Healthcare');
    
    // Navigate away
    await page.click('[data-testid="dashboard-nav"]');
    
    // Come back to form
    await page.click('[data-testid="create-valuation-button"]');
    
    // Verify form data is restored
    await expect(page.locator('[data-testid="company-name-input"]')).toHaveValue('Draft Company');
    await expect(page.locator('[data-testid="abn-input"]')).toHaveValue('98765432109');
    await expect(page.locator('[data-testid="industry-select"]')).toHaveValue('Healthcare');
    
    // Verify draft indicator is shown
    await expect(page.locator('[data-testid="draft-indicator"]')).toBeVisible();
    
    // Clear draft
    await page.click('[data-testid="clear-draft-button"]');
    
    // Verify form is cleared
    await expect(page.locator('[data-testid="company-name-input"]')).toHaveValue('');
    await expect(page.locator('[data-testid="draft-indicator"]')).not.toBeVisible();
  });

  test('mobile responsive design and touch interactions', async ({ page, browserName }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    await loginUser(page);
    
    // Verify mobile navigation
    await expect(page.locator('[data-testid="mobile-menu-button"]')).toBeVisible();
    await page.click('[data-testid="mobile-menu-button"]');
    await expect(page.locator('[data-testid="mobile-menu"]')).toBeVisible();
    
    // Navigate to create valuation from mobile menu
    await page.click('[data-testid="mobile-menu-create-valuation"]');
    await expect(page.locator('[data-testid="mobile-menu"]')).not.toBeVisible();
    
    // Verify form layout on mobile
    await expect(page.locator('[data-testid="valuation-form"]')).toHaveClass(/mobile-layout/);
    
    // Test mobile form interactions
    await page.tap('[data-testid="company-name-input"]');
    await page.type('[data-testid="company-name-input"]', testCompany.name);
    
    // Test dropdown interactions on mobile
    await page.tap('[data-testid="industry-select"]');
    await expect(page.locator('[data-testid="industry-options"]')).toBeVisible();
    await page.tap('[data-testid="industry-technology"]');
    
    // Test date picker on mobile
    await page.tap('[data-testid="ipo-date-input"]');
    if (browserName === 'webkit') {
      // iOS date picker behavior
      await expect(page.locator('[data-testid="ios-date-picker"]')).toBeVisible();
    }
    
    // Verify mobile-optimized validation messages
    await page.tap('[data-testid="submit-valuation-button"]');
    
    const errorMessage = page.locator('[data-testid="error-shares"]');
    await expect(errorMessage).toBeVisible();
    await expect(errorMessage).toHaveClass(/mobile-error-message/);
  });

  test('keyboard navigation and accessibility', async ({ page }) => {
    await loginUser(page);
    
    await page.click('[data-testid="create-valuation-button"]');
    
    // Test keyboard navigation through form
    await page.press('[data-testid="company-name-input"]', 'Tab');
    await expect(page.locator('[data-testid="abn-input"]')).toBeFocused();
    
    await page.press('[data-testid="abn-input"]', 'Tab');
    await expect(page.locator('[data-testid="industry-select"]')).toBeFocused();
    
    // Test dropdown navigation with keyboard
    await page.press('[data-testid="industry-select"]', 'Space');
    await page.press('[data-testid="industry-select"]', 'ArrowDown');
    await page.press('[data-testid="industry-select"]', 'ArrowDown');
    await page.press('[data-testid="industry-select"]', 'Enter');
    
    // Verify selection was made
    await expect(page.locator('[data-testid="industry-select"]')).not.toHaveValue('');
    
    // Test form submission with Enter key
    await page.press('[data-testid="company-name-input"]', 'Enter');
    
    // Should not submit if required fields are empty
    await expect(page.locator('[data-testid="error-messages"]')).toBeVisible();
    
    // Test escape key to close modals
    await page.press('[data-testid="customize-peers-button"]', 'Enter');
    await expect(page.locator('[data-testid="peer-customization-modal"]')).toBeVisible();
    await page.press('body', 'Escape');
    await expect(page.locator('[data-testid="peer-customization-modal"]')).not.toBeVisible();
  });
});