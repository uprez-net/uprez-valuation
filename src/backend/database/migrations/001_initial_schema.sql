-- =====================================================
-- Migration 001: Initial Schema Setup
-- IPO Valuation Platform Database Migration
-- =====================================================

-- Migration metadata
INSERT INTO schema_migrations (version, description, applied_at) 
VALUES ('001', 'Initial schema setup with core tables', NOW());

BEGIN;

-- =====================================================
-- EXTENSIONS SETUP
-- =====================================================

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- =====================================================
-- SCHEMA CREATION
-- =====================================================

-- Create schemas for organization
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS security;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS collaboration;

-- Grant appropriate permissions
GRANT USAGE ON SCHEMA core TO uprez_app_user;
GRANT USAGE ON SCHEMA security TO uprez_app_user;
GRANT USAGE ON SCHEMA analytics TO uprez_analytics_user;
GRANT USAGE ON SCHEMA collaboration TO uprez_app_user;

-- =====================================================
-- CORE TABLES CREATION
-- =====================================================

-- Users table with comprehensive profile management
CREATE TABLE core.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    display_name VARCHAR(200),
    avatar_url TEXT,
    phone VARCHAR(20),
    timezone VARCHAR(50) DEFAULT 'UTC',
    locale VARCHAR(10) DEFAULT 'en-US',
    
    -- Professional information
    title VARCHAR(200),
    organization VARCHAR(300),
    department VARCHAR(200),
    
    -- Authentication
    email_verified BOOLEAN DEFAULT FALSE,
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    two_factor_secret TEXT,
    last_login_at TIMESTAMPTZ,
    login_count INTEGER DEFAULT 0,
    
    -- Account status
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended', 'pending')),
    subscription_tier VARCHAR(20) DEFAULT 'free' CHECK (subscription_tier IN ('free', 'professional', 'enterprise')),
    
    -- Preferences
    preferences JSONB DEFAULT '{}',
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID,
    updated_by UUID
);

-- Organizations table for multi-tenancy
CREATE TABLE core.organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    logo_url TEXT,
    website VARCHAR(500),
    
    -- Contact information
    email VARCHAR(255),
    phone VARCHAR(20),
    address_line1 VARCHAR(500),
    address_line2 VARCHAR(500),
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(2),
    postal_code VARCHAR(20),
    
    -- Business details
    industry VARCHAR(100),
    employee_count INTEGER,
    annual_revenue DECIMAL(15,2),
    
    -- Subscription and billing
    subscription_tier VARCHAR(20) DEFAULT 'free' CHECK (subscription_tier IN ('free', 'professional', 'enterprise')),
    billing_email VARCHAR(255),
    tax_id VARCHAR(100),
    
    -- Status and settings
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended')),
    settings JSONB DEFAULT '{}',
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID,
    updated_by UUID
);

-- User organization memberships
CREATE TABLE core.user_organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    organization_id UUID NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'member' CHECK (role IN ('owner', 'admin', 'manager', 'analyst', 'member', 'viewer')),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'invited', 'rejected')),
    invited_by UUID,
    joined_at TIMESTAMPTZ,
    
    -- Permissions
    permissions JSONB DEFAULT '{}',
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id, organization_id)
);

-- Companies table - core business entities
CREATE TABLE core.companies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Basic company information
    name VARCHAR(300) NOT NULL,
    legal_name VARCHAR(300),
    ticker_symbol VARCHAR(10),
    exchange VARCHAR(20),
    isin VARCHAR(12),
    cusip VARCHAR(9),
    lei VARCHAR(20), -- Legal Entity Identifier
    
    -- Company classification
    sector VARCHAR(100),
    industry VARCHAR(200),
    sub_industry VARCHAR(300),
    gics_sector_code VARCHAR(10),
    gics_industry_code VARCHAR(10),
    
    -- Geographic information
    country_of_incorporation VARCHAR(2),
    country_of_domicile VARCHAR(2),
    headquarters_country VARCHAR(2),
    headquarters_city VARCHAR(100),
    headquarters_state VARCHAR(100),
    
    -- IPO information
    ipo_date DATE,
    ipo_price DECIMAL(12,4),
    ipo_shares_offered BIGINT,
    ipo_proceeds DECIMAL(15,2),
    underwriters TEXT[], -- Array of underwriter names
    
    -- Business description
    description TEXT,
    business_model TEXT,
    competitive_advantages TEXT,
    risk_factors TEXT,
    
    -- Financial year information
    fiscal_year_end DATE,
    reporting_currency VARCHAR(3) DEFAULT 'USD',
    
    -- Data sources and quality
    data_sources TEXT[],
    data_quality_score DECIMAL(3,2),
    last_data_update TIMESTAMPTZ,
    
    -- Status and metadata
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'delisted', 'merged', 'acquired')),
    is_public BOOLEAN DEFAULT FALSE,
    is_ipo_candidate BOOLEAN DEFAULT FALSE,
    
    -- External references
    bloomberg_ticker VARCHAR(50),
    reuters_ric VARCHAR(50),
    factset_entity_id VARCHAR(50),
    
    -- Organization association
    organization_id UUID,
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID,
    updated_by UUID
);

-- Financial statements table
CREATE TABLE core.financial_statements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID NOT NULL,
    
    -- Statement metadata
    statement_type VARCHAR(20) NOT NULL CHECK (statement_type IN ('income_statement', 'balance_sheet', 'cash_flow', 'equity')),
    period_type VARCHAR(20) NOT NULL CHECK (period_type IN ('annual', 'quarterly', 'ttm')),
    period_end_date DATE NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_quarter INTEGER CHECK (fiscal_quarter BETWEEN 1 AND 4),
    
    -- Reporting details
    filing_date DATE,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    scale_factor INTEGER DEFAULT 1, -- 1 = actual, 1000 = thousands, 1000000 = millions
    
    -- Financial data (stored as JSONB for flexibility)
    line_items JSONB NOT NULL,
    
    -- Data source and quality
    source VARCHAR(100),
    source_document_url TEXT,
    data_quality_score DECIMAL(3,2),
    is_restated BOOLEAN DEFAULT FALSE,
    restatement_reason TEXT,
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID,
    updated_by UUID,
    
    UNIQUE(company_id, statement_type, period_type, period_end_date)
);

-- Valuation models table
CREATE TABLE core.valuation_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    
    -- Model categorization
    model_type VARCHAR(50) NOT NULL CHECK (model_type IN ('dcf', 'comparable_company', 'precedent_transaction', 'asset_based', 'sum_of_parts', 'option_pricing', 'monte_carlo')),
    model_category VARCHAR(50) CHECK (model_category IN ('intrinsic', 'relative', 'hybrid')),
    
    -- Model configuration
    methodology JSONB NOT NULL, -- Stores model-specific parameters
    assumptions JSONB, -- Key assumptions and inputs
    sensitivity_parameters JSONB, -- Parameters for sensitivity analysis
    
    -- Model metadata
    version VARCHAR(20) DEFAULT '1.0',
    is_template BOOLEAN DEFAULT FALSE,
    is_public BOOLEAN DEFAULT FALSE,
    complexity_level VARCHAR(20) CHECK (complexity_level IN ('basic', 'intermediate', 'advanced')),
    
    -- Usage and performance
    usage_count INTEGER DEFAULT 0,
    average_rating DECIMAL(3,2),
    
    -- Organization and ownership
    organization_id UUID,
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID,
    updated_by UUID
);

-- Valuations table - stores valuation results
CREATE TABLE core.valuations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID NOT NULL,
    model_id UUID NOT NULL,
    
    -- Valuation metadata
    valuation_name VARCHAR(200) NOT NULL,
    valuation_purpose TEXT,
    valuation_date DATE NOT NULL DEFAULT CURRENT_DATE,
    base_date DATE NOT NULL, -- Date as of which valuation is performed
    
    -- Model inputs and assumptions
    inputs JSONB NOT NULL,
    assumptions JSONB NOT NULL,
    scenario VARCHAR(50) DEFAULT 'base' CHECK (scenario IN ('bear', 'base', 'bull', 'stress')),
    
    -- Valuation results
    enterprise_value DECIMAL(15,2),
    equity_value DECIMAL(15,2),
    per_share_value DECIMAL(12,4),
    
    -- Valuation ranges and sensitivity
    value_range_low DECIMAL(12,4),
    value_range_high DECIMAL(12,4),
    confidence_level DECIMAL(3,2), -- 0.00 to 1.00
    
    -- Supporting calculations
    calculations JSONB, -- Detailed calculation steps
    financial_projections JSONB, -- Multi-year projections
    
    -- Analysis and insights
    key_value_drivers TEXT[],
    sensitivity_analysis JSONB,
    risk_assessment JSONB,
    
    -- Comparison and benchmarking
    peer_comparison JSONB,
    market_multiples JSONB,
    
    -- Status and workflow
    status VARCHAR(20) DEFAULT 'draft' CHECK (status IN ('draft', 'under_review', 'approved', 'final', 'archived')),
    review_notes TEXT,
    approved_by UUID,
    approved_at TIMESTAMPTZ,
    
    -- Version control
    version VARCHAR(20) DEFAULT '1.0',
    parent_valuation_id UUID,
    
    -- Organization and access
    organization_id UUID,
    is_shared BOOLEAN DEFAULT FALSE,
    access_level VARCHAR(20) DEFAULT 'private' CHECK (access_level IN ('private', 'organization', 'public')),
    
    -- Tags and categorization
    tags TEXT[],
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID,
    updated_by UUID
);

-- =====================================================
-- FOREIGN KEY CONSTRAINTS
-- =====================================================

-- Add foreign key constraints after all tables are created
ALTER TABLE core.users ADD CONSTRAINT fk_users_created_by FOREIGN KEY (created_by) REFERENCES core.users(id);
ALTER TABLE core.users ADD CONSTRAINT fk_users_updated_by FOREIGN KEY (updated_by) REFERENCES core.users(id);

ALTER TABLE core.organizations ADD CONSTRAINT fk_organizations_created_by FOREIGN KEY (created_by) REFERENCES core.users(id);
ALTER TABLE core.organizations ADD CONSTRAINT fk_organizations_updated_by FOREIGN KEY (updated_by) REFERENCES core.users(id);

ALTER TABLE core.user_organizations ADD CONSTRAINT fk_user_organizations_user FOREIGN KEY (user_id) REFERENCES core.users(id) ON DELETE CASCADE;
ALTER TABLE core.user_organizations ADD CONSTRAINT fk_user_organizations_org FOREIGN KEY (organization_id) REFERENCES core.organizations(id) ON DELETE CASCADE;
ALTER TABLE core.user_organizations ADD CONSTRAINT fk_user_organizations_invited_by FOREIGN KEY (invited_by) REFERENCES core.users(id);

ALTER TABLE core.companies ADD CONSTRAINT fk_companies_organization FOREIGN KEY (organization_id) REFERENCES core.organizations(id);
ALTER TABLE core.companies ADD CONSTRAINT fk_companies_created_by FOREIGN KEY (created_by) REFERENCES core.users(id);
ALTER TABLE core.companies ADD CONSTRAINT fk_companies_updated_by FOREIGN KEY (updated_by) REFERENCES core.users(id);

ALTER TABLE core.financial_statements ADD CONSTRAINT fk_financial_statements_company FOREIGN KEY (company_id) REFERENCES core.companies(id) ON DELETE CASCADE;
ALTER TABLE core.financial_statements ADD CONSTRAINT fk_financial_statements_created_by FOREIGN KEY (created_by) REFERENCES core.users(id);
ALTER TABLE core.financial_statements ADD CONSTRAINT fk_financial_statements_updated_by FOREIGN KEY (updated_by) REFERENCES core.users(id);

ALTER TABLE core.valuation_models ADD CONSTRAINT fk_valuation_models_organization FOREIGN KEY (organization_id) REFERENCES core.organizations(id);
ALTER TABLE core.valuation_models ADD CONSTRAINT fk_valuation_models_created_by FOREIGN KEY (created_by) REFERENCES core.users(id);
ALTER TABLE core.valuation_models ADD CONSTRAINT fk_valuation_models_updated_by FOREIGN KEY (updated_by) REFERENCES core.users(id);

ALTER TABLE core.valuations ADD CONSTRAINT fk_valuations_company FOREIGN KEY (company_id) REFERENCES core.companies(id) ON DELETE CASCADE;
ALTER TABLE core.valuations ADD CONSTRAINT fk_valuations_model FOREIGN KEY (model_id) REFERENCES core.valuation_models(id);
ALTER TABLE core.valuations ADD CONSTRAINT fk_valuations_approved_by FOREIGN KEY (approved_by) REFERENCES core.users(id);
ALTER TABLE core.valuations ADD CONSTRAINT fk_valuations_parent FOREIGN KEY (parent_valuation_id) REFERENCES core.valuations(id);
ALTER TABLE core.valuations ADD CONSTRAINT fk_valuations_organization FOREIGN KEY (organization_id) REFERENCES core.organizations(id);
ALTER TABLE core.valuations ADD CONSTRAINT fk_valuations_created_by FOREIGN KEY (created_by) REFERENCES core.users(id);
ALTER TABLE core.valuations ADD CONSTRAINT fk_valuations_updated_by FOREIGN KEY (updated_by) REFERENCES core.users(id);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Users table indexes
CREATE INDEX idx_users_email ON core.users (email);
CREATE INDEX idx_users_status ON core.users (status);
CREATE INDEX idx_users_org_lookup ON core.users USING gin (email, display_name);
CREATE INDEX idx_users_created_at ON core.users (created_at);

-- Organizations table indexes
CREATE INDEX idx_organizations_slug ON core.organizations (slug);
CREATE INDEX idx_organizations_status ON core.organizations (status);
CREATE INDEX idx_organizations_industry ON core.organizations (industry);

-- User organizations table indexes
CREATE INDEX idx_user_orgs_user_id ON core.user_organizations (user_id);
CREATE INDEX idx_user_orgs_org_id ON core.user_organizations (organization_id);
CREATE INDEX idx_user_orgs_role ON core.user_organizations (role);

-- Companies table indexes
CREATE INDEX idx_companies_ticker ON core.companies (ticker_symbol);
CREATE INDEX idx_companies_name ON core.companies USING gin (name gin_trgm_ops);
CREATE INDEX idx_companies_sector_industry ON core.companies (sector, industry);
CREATE INDEX idx_companies_country ON core.companies (country_of_incorporation);
CREATE INDEX idx_companies_ipo_date ON core.companies (ipo_date);
CREATE INDEX idx_companies_status ON core.companies (status);
CREATE INDEX idx_companies_org_id ON core.companies (organization_id);
CREATE INDEX idx_companies_external_ids ON core.companies USING gin (bloomberg_ticker, reuters_ric, factset_entity_id);

-- Financial statements table indexes
CREATE INDEX idx_financial_statements_company_id ON core.financial_statements (company_id);
CREATE INDEX idx_financial_statements_period ON core.financial_statements (period_end_date DESC);
CREATE INDEX idx_financial_statements_type_period ON core.financial_statements (statement_type, period_type, period_end_date DESC);
CREATE INDEX idx_financial_statements_fiscal_year ON core.financial_statements (fiscal_year DESC);
CREATE INDEX idx_financial_statements_line_items ON core.financial_statements USING gin (line_items);

-- Valuation models table indexes
CREATE INDEX idx_valuation_models_type ON core.valuation_models (model_type);
CREATE INDEX idx_valuation_models_org_id ON core.valuation_models (organization_id);
CREATE INDEX idx_valuation_models_public ON core.valuation_models (is_public);
CREATE INDEX idx_valuation_models_template ON core.valuation_models (is_template);

-- Valuations table indexes
CREATE INDEX idx_valuations_company_id ON core.valuations (company_id);
CREATE INDEX idx_valuations_model_id ON core.valuations (model_id);
CREATE INDEX idx_valuations_date ON core.valuations (valuation_date DESC);
CREATE INDEX idx_valuations_status ON core.valuations (status);
CREATE INDEX idx_valuations_org_id ON core.valuations (organization_id);
CREATE INDEX idx_valuations_created_by ON core.valuations (created_by);
CREATE INDEX idx_valuations_tags ON core.valuations USING gin (tags);
CREATE INDEX idx_valuations_per_share_value ON core.valuations (per_share_value);

-- Composite indexes for common queries
CREATE INDEX idx_companies_sector_ipo ON core.companies (sector, ipo_date DESC) WHERE status = 'active';
CREATE INDEX idx_valuations_company_date ON core.valuations (company_id, valuation_date DESC);
CREATE INDEX idx_financial_statements_company_type_date ON core.financial_statements (company_id, statement_type, period_end_date DESC);

-- =====================================================
-- TRIGGERS FOR AUDIT TRAIL
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE 'plpgsql';

-- Apply update triggers to all main tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON core.users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON core.organizations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_organizations_updated_at BEFORE UPDATE ON core.user_organizations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_companies_updated_at BEFORE UPDATE ON core.companies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_financial_statements_updated_at BEFORE UPDATE ON core.financial_statements FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_valuation_models_updated_at BEFORE UPDATE ON core.valuation_models FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_valuations_updated_at BEFORE UPDATE ON core.valuations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- PERMISSIONS AND ROLES
-- =====================================================

-- Create database roles
CREATE ROLE uprez_app_user;
CREATE ROLE uprez_analytics_user;
CREATE ROLE uprez_etl_user;

-- Grant permissions to application user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA core TO uprez_app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA collaboration TO uprez_app_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA core TO uprez_app_user;

-- Grant permissions to analytics user (read-only)
GRANT SELECT ON ALL TABLES IN SCHEMA core TO uprez_analytics_user;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO uprez_analytics_user;

-- Grant permissions to ETL user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA core TO uprez_etl_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA analytics TO uprez_etl_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA core TO uprez_etl_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA analytics TO uprez_etl_user;

-- =====================================================
-- DEFAULT DATA
-- =====================================================

-- Insert default valuation model templates
INSERT INTO core.valuation_models (name, description, model_type, model_category, methodology, is_template, is_public, complexity_level) VALUES
('Basic DCF Model', 'Simple discounted cash flow model for quick valuations', 'dcf', 'intrinsic', 
 '{"discount_rate": "wacc", "terminal_growth_rate": 0.03, "forecast_years": 5, "terminal_method": "perpetual_growth"}', 
 true, true, 'basic'),
 
('Comparable Company Analysis', 'Valuation based on trading multiples of peer companies', 'comparable_company', 'relative',
 '{"multiples": ["pe_ratio", "ev_ebitda", "ev_revenue"], "peer_selection": "sector_industry", "adjustments": ["size", "growth", "profitability"]}',
 true, true, 'intermediate'),
 
('Precedent Transaction Analysis', 'Valuation based on acquisition multiples from similar transactions', 'precedent_transaction', 'relative',
 '{"multiples": ["ev_ebitda", "ev_revenue"], "transaction_filters": ["same_industry", "similar_size"], "premium_analysis": true}',
 true, true, 'intermediate'),
 
('Monte Carlo Simulation', 'Advanced probabilistic valuation using Monte Carlo methods', 'monte_carlo', 'hybrid',
 '{"iterations": 10000, "key_variables": ["revenue_growth", "margin_expansion", "discount_rate"], "distributions": "triangular"}',
 true, false, 'advanced');

COMMIT;

-- =====================================================
-- MIGRATION COMPLETION
-- =====================================================

-- Update migration status
UPDATE schema_migrations 
SET completed_at = NOW() 
WHERE version = '001';