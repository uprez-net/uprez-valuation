import { defineConfig, devices } from '@playwright/test';
import path from 'path';

// Read environment variables
const baseURL = process.env.BASE_URL || 'http://localhost:3000';
const CI = !!process.env.CI;

/**
 * Playwright configuration for E2E testing of the IPO Valuation Platform.
 * 
 * This configuration includes:
 * - Cross-browser testing (Chrome, Firefox, Safari)
 * - Mobile device simulation
 * - Visual regression testing
 * - Accessibility testing
 * - Performance monitoring
 */
export default defineConfig({
  testDir: './e2e',
  outputDir: './test-results',
  
  // Global test timeout
  timeout: 30 * 1000,
  expect: {
    // Global assertion timeout
    timeout: 5 * 1000,
    // Visual comparison threshold
    threshold: 0.2,
  },

  // Test execution settings
  fullyParallel: !CI, // Run tests in parallel locally, sequentially in CI
  forbidOnly: CI, // Prevent accidentally committed test.only
  retries: CI ? 2 : 0, // Retry failed tests in CI
  workers: CI ? 1 : undefined, // Use single worker in CI for stability
  
  // Reporter configuration
  reporter: [
    ['html', { outputFolder: 'playwright-report', open: !CI }],
    ['json', { outputFile: 'test-results/results.json' }],
    ['junit', { outputFile: 'test-results/junit.xml' }],
    ...(CI ? [['github']] : [['list']]),
  ],

  // Global test configuration
  use: {
    // Base URL for navigation
    baseURL,
    
    // Browser context options
    locale: 'en-AU',
    timezoneId: 'Australia/Sydney',
    
    // Tracing and debugging
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    
    // Network and performance
    actionTimeout: 10 * 1000,
    navigationTimeout: 15 * 1000,
    
    // Test data
    extraHTTPHeaders: {
      'Accept-Language': 'en-AU',
    },
  },

  // Test projects for different browsers and devices
  projects: [
    // Desktop browsers
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },

    // Mobile devices
    {
      name: 'mobile-chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'mobile-safari',
      use: { ...devices['iPhone 12'] },
    },

    // Tablet
    {
      name: 'tablet',
      use: { ...devices['iPad Pro'] },
    },

    // Accessibility testing
    {
      name: 'accessibility',
      testMatch: '**/accessibility/**/*.spec.ts',
      use: { ...devices['Desktop Chrome'] },
    },

    // Visual regression testing
    {
      name: 'visual',
      testMatch: '**/visual/**/*.spec.ts',
      use: {
        ...devices['Desktop Chrome'],
        // Consistent viewport for visual tests
        viewport: { width: 1280, height: 720 },
      },
    },

    // Performance testing
    {
      name: 'performance',
      testMatch: '**/performance/**/*.spec.ts',
      use: {
        ...devices['Desktop Chrome'],
        // Enable performance metrics
        launchOptions: {
          args: ['--enable-precise-memory-info'],
        },
      },
    },
  ],

  // Global setup and teardown
  globalSetup: path.resolve(__dirname, 'global-setup.ts'),
  globalTeardown: path.resolve(__dirname, 'global-teardown.ts'),

  // Web server for development testing
  webServer: process.env.CI
    ? undefined
    : {
        command: 'npm run serve:test',
        port: 3001,
        reuseExistingServer: !CI,
        timeout: 60 * 1000,
      },

  // Metadata for reports
  metadata: {
    'test-type': 'e2e',
    'app': 'IPO Valuation Platform',
    'environment': process.env.NODE_ENV || 'test',
    'base-url': baseURL,
  },
});