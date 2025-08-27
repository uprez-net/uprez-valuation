# IPO Valuation Platform - Testing Makefile
# 
# Comprehensive test automation for all test categories including:
# - Backend testing (unit, integration, API)
# - Frontend testing (components, e2e, visual)
# - ML/AI testing (accuracy, drift, performance)
# - Performance testing (load, stress, scalability)
# - Security testing (vulnerabilities, penetration)
# - Integration testing (external services, databases)

.PHONY: help test test-all test-backend test-frontend test-ml test-integration test-performance test-security
.PHONY: test-coverage test-ci test-local test-docker test-smoke
.PHONY: lint format clean install-deps setup-test-env
.PHONY: test-reports test-benchmark test-quality-gates

# Default target
help: ## Show this help message
	@echo "IPO Valuation Platform - Testing Suite"
	@echo "======================================"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\033[36m%-25s\033[0m %s\n", "Target", "Description"} /^[a-zA-Z_-]+:.*?##/ { printf "\033[36m%-25s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make test              # Run all tests"
	@echo "  make test-backend      # Run backend tests only"
	@echo "  make test-coverage     # Run tests with coverage"
	@echo "  make test-ci           # Run tests for CI environment"
	@echo ""

# Variables
PYTHON := python3
PIP := pip3
NODE := node
NPM := npm
DOCKER := docker
DOCKER_COMPOSE := docker-compose

# Directories
BACKEND_DIR := ../src/backend
FRONTEND_DIR := ./frontend
TEST_DIR := .
REPORTS_DIR := ./reports
COVERAGE_DIR := ./coverage

# Test configuration
PYTEST_ARGS := -v --tb=short
JEST_ARGS := --verbose --colors
PLAYWRIGHT_ARGS := --reporter=html
COVERAGE_THRESHOLD := 80
MAX_WORKERS := auto

##@ Setup and Installation

install-deps: ## Install all test dependencies
	@echo "Installing Python test dependencies..."
	$(PIP) install -r requirements-test.txt
	@echo "Installing Node.js test dependencies..."
	cd $(FRONTEND_DIR) && $(NPM) install
	@echo "Installing Playwright browsers..."
	cd $(FRONTEND_DIR) && npx playwright install --with-deps
	@echo "Dependencies installed successfully!"

setup-test-env: ## Setup test environment and databases
	@echo "Setting up test environment..."
	@mkdir -p $(REPORTS_DIR) $(COVERAGE_DIR)
	@echo "Creating test databases..."
	docker-compose -f docker-compose.test.yml up -d postgres redis
	@echo "Waiting for databases to be ready..."
	@sleep 10
	@echo "Running database migrations..."
	cd $(BACKEND_DIR) && alembic upgrade head
	@echo "Test environment setup complete!"

clean: ## Clean test artifacts and temporary files
	@echo "Cleaning test artifacts..."
	@rm -rf $(REPORTS_DIR)/* $(COVERAGE_DIR)/*
	@rm -rf **/__pycache__ **/*.pyc **/*.pyo
	@rm -rf .pytest_cache .coverage
	@rm -rf $(FRONTEND_DIR)/coverage $(FRONTEND_DIR)/test-results
	@rm -rf $(FRONTEND_DIR)/playwright-report
	@echo "Cleanup complete!"

##@ Core Testing

test: test-all ## Run all tests (alias for test-all)

test-all: ## Run complete test suite
	@echo "Running complete test suite..."
	$(MAKE) test-backend
	$(MAKE) test-frontend
	$(MAKE) test-integration
	$(MAKE) test-performance
	$(MAKE) test-security
	@echo "All tests completed!"

test-local: ## Run tests suitable for local development
	@echo "Running local development tests..."
	$(MAKE) test-backend-unit
	$(MAKE) test-frontend-unit
	$(MAKE) test-integration-basic
	@echo "Local tests completed!"

test-smoke: ## Run smoke tests (quick validation)
	@echo "Running smoke tests..."
	pytest $(PYTEST_ARGS) -m "not slow and not external" tests/backend/unit/ -x --maxfail=5
	cd $(FRONTEND_DIR) && npm run test -- --testPathPattern="smoke" --passWithNoTests
	@echo "Smoke tests completed!"

test-ci: ## Run tests optimized for CI environment
	@echo "Running CI test suite..."
	@export CI=true
	$(MAKE) test-backend-ci
	$(MAKE) test-frontend-ci
	$(MAKE) test-integration-ci
	$(MAKE) test-performance-ci
	$(MAKE) test-security-ci
	@echo "CI tests completed!"

##@ Backend Testing

test-backend: ## Run all backend tests
	@echo "Running backend test suite..."
	$(MAKE) test-backend-unit
	$(MAKE) test-backend-integration
	$(MAKE) test-backend-api
	$(MAKE) test-backend-ml

test-backend-unit: ## Run backend unit tests
	@echo "Running backend unit tests..."
	pytest $(PYTEST_ARGS) tests/backend/unit/ \
		--cov=src/backend \
		--cov-report=html:$(COVERAGE_DIR)/backend-unit \
		--cov-report=xml:$(REPORTS_DIR)/backend-unit-coverage.xml \
		--junit-xml=$(REPORTS_DIR)/backend-unit-results.xml

test-backend-integration: ## Run backend integration tests
	@echo "Running backend integration tests..."
	pytest $(PYTEST_ARGS) tests/backend/integration/ \
		--cov=src/backend \
		--cov-report=html:$(COVERAGE_DIR)/backend-integration \
		--junit-xml=$(REPORTS_DIR)/backend-integration-results.xml

test-backend-api: ## Run API endpoint tests
	@echo "Running API endpoint tests..."
	pytest $(PYTEST_ARGS) tests/backend/api/ \
		--cov=src/backend/api \
		--junit-xml=$(REPORTS_DIR)/backend-api-results.xml

test-backend-ml: ## Run ML/AI model tests
	@echo "Running ML/AI model tests..."
	pytest $(PYTEST_ARGS) tests/backend/ml/ \
		-m "not slow" \
		--junit-xml=$(REPORTS_DIR)/backend-ml-results.xml

test-backend-ml-full: ## Run complete ML tests including slow tests
	@echo "Running complete ML/AI test suite..."
	pytest $(PYTEST_ARGS) tests/backend/ml/ \
		--junit-xml=$(REPORTS_DIR)/backend-ml-full-results.xml

test-backend-ci: ## Run backend tests for CI
	@echo "Running backend tests for CI..."
	pytest $(PYTEST_ARGS) tests/backend/ \
		--cov=src/backend \
		--cov-report=xml:$(REPORTS_DIR)/backend-coverage.xml \
		--cov-report=html:$(COVERAGE_DIR)/backend \
		--junit-xml=$(REPORTS_DIR)/backend-results.xml \
		--cov-fail-under=$(COVERAGE_THRESHOLD) \
		-n $(MAX_WORKERS) \
		-x --maxfail=10

##@ Frontend Testing

test-frontend: ## Run all frontend tests
	@echo "Running frontend test suite..."
	$(MAKE) test-frontend-unit
	$(MAKE) test-frontend-e2e
	$(MAKE) test-frontend-visual

test-frontend-unit: ## Run frontend unit/component tests
	@echo "Running frontend unit tests..."
	cd $(FRONTEND_DIR) && npm run test:coverage

test-frontend-e2e: ## Run end-to-end tests
	@echo "Running E2E tests..."
	cd $(FRONTEND_DIR) && npm run test:e2e

test-frontend-e2e-headed: ## Run E2E tests in headed mode (for debugging)
	@echo "Running E2E tests in headed mode..."
	cd $(FRONTEND_DIR) && npm run test:e2e:headed

test-frontend-visual: ## Run visual regression tests
	@echo "Running visual regression tests..."
	cd $(FRONTEND_DIR) && npm run test:visual

test-frontend-accessibility: ## Run accessibility tests
	@echo "Running accessibility tests..."
	cd $(FRONTEND_DIR) && npm run test:accessibility

test-frontend-performance: ## Run frontend performance tests
	@echo "Running frontend performance tests..."
	cd $(FRONTEND_DIR) && npm run test:performance

test-frontend-ci: ## Run frontend tests for CI
	@echo "Running frontend tests for CI..."
	cd $(FRONTEND_DIR) && npm run test:ci
	cd $(FRONTEND_DIR) && npm run test:e2e -- --reporter=junit

##@ ML/AI Testing

test-ml: ## Run ML/AI tests
	@echo "Running ML/AI test suite..."
	pytest $(PYTEST_ARGS) tests/backend/ml/ \
		--junit-xml=$(REPORTS_DIR)/ml-results.xml

test-ml-accuracy: ## Run model accuracy validation tests
	@echo "Running model accuracy tests..."
	pytest $(PYTEST_ARGS) tests/backend/ml/test_model_accuracy.py \
		-v --tb=long

test-ml-drift: ## Run data drift detection tests
	@echo "Running data drift detection tests..."
	pytest $(PYTEST_ARGS) tests/backend/ml/ \
		-k "drift" -v

test-ml-performance: ## Run ML model performance tests
	@echo "Running ML performance tests..."
	pytest $(PYTEST_ARGS) tests/backend/ml/ \
		-m "performance" -v

test-ml-benchmark: ## Run ML model benchmarking
	@echo "Running ML model benchmarks..."
	pytest $(PYTEST_ARGS) tests/backend/ml/ \
		--benchmark-only \
		--benchmark-json=$(REPORTS_DIR)/ml-benchmarks.json

##@ Integration Testing

test-integration: ## Run integration tests
	@echo "Running integration test suite..."
	$(MAKE) test-integration-basic
	$(MAKE) test-integration-external
	$(MAKE) test-integration-e2e

test-integration-basic: ## Run basic integration tests
	@echo "Running basic integration tests..."
	pytest $(PYTEST_ARGS) tests/integration/ \
		-m "not external" \
		--junit-xml=$(REPORTS_DIR)/integration-basic-results.xml

test-integration-external: ## Run external service integration tests
	@echo "Running external service integration tests..."
	pytest $(PYTEST_ARGS) tests/integration/ \
		-m "external" \
		--junit-xml=$(REPORTS_DIR)/integration-external-results.xml

test-integration-e2e: ## Run end-to-end integration tests
	@echo "Running E2E integration tests..."
	pytest $(PYTEST_ARGS) tests/integration/ \
		-m "e2e" \
		--junit-xml=$(REPORTS_DIR)/integration-e2e-results.xml

test-integration-ci: ## Run integration tests for CI
	@echo "Running integration tests for CI..."
	pytest $(PYTEST_ARGS) tests/integration/ \
		-m "not slow" \
		-x --maxfail=5 \
		--junit-xml=$(REPORTS_DIR)/integration-results.xml

##@ Performance Testing

test-performance: ## Run performance test suite
	@echo "Running performance test suite..."
	$(MAKE) test-load
	$(MAKE) test-stress
	$(MAKE) test-scalability

test-load: ## Run load testing
	@echo "Running load tests..."
	cd tests/performance && locust -f load_testing.py --headless \
		--users 50 --spawn-rate 5 -t 5m \
		--host http://localhost:8000 \
		--html $(REPORTS_DIR)/load-test-report.html \
		--csv $(REPORTS_DIR)/load-test

test-stress: ## Run stress testing
	@echo "Running stress tests..."
	cd tests/performance && locust -f load_testing.py --headless \
		--users 200 --spawn-rate 10 -t 10m \
		--host http://localhost:8000 \
		--html $(REPORTS_DIR)/stress-test-report.html \
		--csv $(REPORTS_DIR)/stress-test

test-spike: ## Run spike testing
	@echo "Running spike tests..."
	cd tests/performance && locust -f load_testing.py --headless \
		--users 500 --spawn-rate 50 -t 3m \
		--host http://localhost:8000 \
		--html $(REPORTS_DIR)/spike-test-report.html

test-endurance: ## Run endurance testing
	@echo "Running endurance tests..."
	cd tests/performance && locust -f load_testing.py --headless \
		--users 100 --spawn-rate 5 -t 60m \
		--host http://localhost:8000 \
		--html $(REPORTS_DIR)/endurance-test-report.html

test-scalability: ## Run scalability testing
	@echo "Running scalability tests..."
	pytest $(PYTEST_ARGS) tests/performance/test_scalability.py \
		--junit-xml=$(REPORTS_DIR)/scalability-results.xml

test-performance-ci: ## Run performance tests for CI
	@echo "Running performance tests for CI (light)..."
	cd tests/performance && locust -f load_testing.py --headless \
		--users 20 --spawn-rate 5 -t 2m \
		--host http://localhost:8000 \
		--html $(REPORTS_DIR)/ci-performance-report.html

##@ Security Testing

test-security: ## Run security test suite
	@echo "Running security test suite..."
	$(MAKE) test-security-vulnerabilities
	$(MAKE) test-security-auth
	$(MAKE) test-security-data

test-security-vulnerabilities: ## Run vulnerability tests
	@echo "Running security vulnerability tests..."
	pytest $(PYTEST_ARGS) tests/security/test_security_vulnerabilities.py \
		--junit-xml=$(REPORTS_DIR)/security-vulnerabilities-results.xml

test-security-auth: ## Run authentication/authorization tests
	@echo "Running authentication security tests..."
	pytest $(PYTEST_ARGS) tests/security/ \
		-k "auth" \
		--junit-xml=$(REPORTS_DIR)/security-auth-results.xml

test-security-data: ## Run data protection security tests
	@echo "Running data protection tests..."
	pytest $(PYTEST_ARGS) tests/security/ \
		-k "data" \
		--junit-xml=$(REPORTS_DIR)/security-data-results.xml

test-security-scan: ## Run security scanning with bandit
	@echo "Running security scan..."
	bandit -r $(BACKEND_DIR) -f json -o $(REPORTS_DIR)/security-scan.json || true
	bandit -r $(BACKEND_DIR) -f html -o $(REPORTS_DIR)/security-scan.html || true

test-security-ci: ## Run security tests for CI
	@echo "Running security tests for CI..."
	pytest $(PYTEST_ARGS) tests/security/ \
		-x --maxfail=5 \
		--junit-xml=$(REPORTS_DIR)/security-results.xml

##@ Coverage and Quality

test-coverage: ## Run tests with coverage analysis
	@echo "Running tests with coverage analysis..."
	pytest $(PYTEST_ARGS) tests/backend/ \
		--cov=src/backend \
		--cov-report=html:$(COVERAGE_DIR)/backend \
		--cov-report=xml:$(REPORTS_DIR)/backend-coverage.xml \
		--cov-report=term-missing \
		--cov-fail-under=$(COVERAGE_THRESHOLD)
	
	cd $(FRONTEND_DIR) && npm run test:coverage
	@echo "Coverage analysis complete!"

test-quality-gates: ## Run quality gate checks
	@echo "Running quality gate checks..."
	# Backend quality checks
	pytest $(PYTEST_ARGS) tests/backend/ \
		--cov=src/backend \
		--cov-fail-under=$(COVERAGE_THRESHOLD) \
		--maxfail=0
	
	# Frontend quality checks
	cd $(FRONTEND_DIR) && npm run test:ci
	cd $(FRONTEND_DIR) && npm run lint
	cd $(FRONTEND_DIR) && npm run typecheck
	
	# Security checks
	$(MAKE) test-security-scan
	
	@echo "Quality gates passed!"

lint: ## Run code linting
	@echo "Running backend linting..."
	flake8 $(BACKEND_DIR)
	black --check $(BACKEND_DIR)
	isort --check-only $(BACKEND_DIR)
	mypy $(BACKEND_DIR)
	
	@echo "Running frontend linting..."
	cd $(FRONTEND_DIR) && npm run lint
	
	@echo "Linting complete!"

format: ## Format code
	@echo "Formatting backend code..."
	black $(BACKEND_DIR)
	isort $(BACKEND_DIR)
	
	@echo "Formatting frontend code..."
	cd $(FRONTEND_DIR) && npm run lint -- --fix
	
	@echo "Code formatting complete!"

##@ Docker Testing

test-docker: ## Run tests in Docker containers
	@echo "Running tests in Docker..."
	$(DOCKER_COMPOSE) -f docker-compose.test.yml up --build --abort-on-container-exit

test-docker-build: ## Build test Docker images
	@echo "Building test Docker images..."
	$(DOCKER_COMPOSE) -f docker-compose.test.yml build

test-docker-cleanup: ## Clean up Docker test resources
	@echo "Cleaning up Docker test resources..."
	$(DOCKER_COMPOSE) -f docker-compose.test.yml down -v --remove-orphans

##@ Reporting and Analysis

test-reports: ## Generate comprehensive test reports
	@echo "Generating test reports..."
	@mkdir -p $(REPORTS_DIR)
	
	# Combine JUnit XML reports
	pytest-html-combine $(REPORTS_DIR)/*-results.xml \
		--output $(REPORTS_DIR)/combined-test-report.html || true
	
	# Generate coverage badge
	coverage-badge -o $(REPORTS_DIR)/coverage-badge.svg || true
	
	@echo "Test reports generated in $(REPORTS_DIR)/"

test-benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	pytest $(PYTEST_ARGS) tests/ \
		--benchmark-only \
		--benchmark-json=$(REPORTS_DIR)/benchmarks.json \
		--benchmark-html=$(REPORTS_DIR)/benchmarks.html

analyze-flaky-tests: ## Analyze flaky test patterns
	@echo "Analyzing flaky test patterns..."
	pytest $(PYTEST_ARGS) tests/ \
		--lf --tb=no -q \
		--json-report --json-report-file=$(REPORTS_DIR)/flaky-analysis.json

##@ Utility Commands

watch-backend: ## Watch and run backend tests on file changes
	@echo "Watching backend tests..."
	ptw -- tests/backend/ $(PYTEST_ARGS)

watch-frontend: ## Watch and run frontend tests on file changes
	@echo "Watching frontend tests..."
	cd $(FRONTEND_DIR) && npm run test:watch

debug-test: ## Run specific test with debugging
	@echo "Debug mode - specify test with TEST_PATH variable"
	@echo "Example: make debug-test TEST_PATH=tests/backend/unit/test_api_main.py::TestApplicationCreation::test_create_application_returns_fastapi_instance"
	pytest $(TEST_PATH) -v --tb=long --pdb

profile-tests: ## Profile test execution performance
	@echo "Profiling test execution..."
	pytest $(PYTEST_ARGS) tests/backend/ \
		--profile --profile-svg

validate-test-data: ## Validate test data integrity
	@echo "Validating test data..."
	python tests/utils/validate_test_data.py

# Environment-specific targets
test-dev: test-local ## Run development tests (alias)
test-staging: test-ci ## Run staging tests (alias) 
test-production: test-smoke ## Run production tests (alias)

# Help for specific categories
help-backend: ## Show backend testing help
	@echo "Backend Testing Commands:"
	@echo "  test-backend          - Run all backend tests"
	@echo "  test-backend-unit     - Unit tests only"
	@echo "  test-backend-api      - API endpoint tests"
	@echo "  test-backend-ml       - ML/AI model tests"

help-frontend: ## Show frontend testing help
	@echo "Frontend Testing Commands:"
	@echo "  test-frontend         - Run all frontend tests"
	@echo "  test-frontend-unit    - Component unit tests"
	@echo "  test-frontend-e2e     - End-to-end tests"
	@echo "  test-frontend-visual  - Visual regression tests"

help-performance: ## Show performance testing help
	@echo "Performance Testing Commands:"
	@echo "  test-performance      - Run all performance tests"
	@echo "  test-load            - Load testing"
	@echo "  test-stress          - Stress testing"
	@echo "  test-spike           - Spike testing"

help-security: ## Show security testing help
	@echo "Security Testing Commands:"
	@echo "  test-security         - Run all security tests"
	@echo "  test-security-scan    - Security vulnerability scan"
	@echo "  test-security-auth    - Authentication tests"

# Default target when no target is specified
.DEFAULT_GOAL := help