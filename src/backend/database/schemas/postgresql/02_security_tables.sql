-- =====================================================
-- Security and Audit Tables
-- =====================================================

-- API keys and tokens for external integrations
CREATE TABLE security.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES core.users(id) ON DELETE CASCADE,
    organization_id UUID REFERENCES core.organizations(id) ON DELETE CASCADE,
    
    -- Key metadata
    name VARCHAR(200) NOT NULL,
    description TEXT,
    key_hash VARCHAR(255) NOT NULL UNIQUE, -- Hashed version of the key
    key_prefix VARCHAR(20) NOT NULL, -- First few characters for identification
    
    -- Permissions and scope
    scopes TEXT[] DEFAULT '{}', -- Array of permitted scopes
    permissions JSONB DEFAULT '{}',
    rate_limit INTEGER DEFAULT 1000, -- Requests per hour
    
    -- Usage tracking
    last_used_at TIMESTAMPTZ,
    usage_count INTEGER DEFAULT 0,
    
    -- Security settings
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    allowed_ips INET[],
    allowed_domains TEXT[],
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID REFERENCES core.users(id)
);

-- User sessions for authentication tracking
CREATE TABLE security.user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES core.users(id) ON DELETE CASCADE,
    
    -- Session data
    session_token_hash VARCHAR(255) NOT NULL UNIQUE,
    refresh_token_hash VARCHAR(255),
    
    -- Device and location information
    user_agent TEXT,
    ip_address INET,
    device_fingerprint VARCHAR(255),
    location_country VARCHAR(2),
    location_city VARCHAR(100),
    
    -- Session lifecycle
    expires_at TIMESTAMPTZ NOT NULL,
    last_activity_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Security flags
    is_suspicious BOOLEAN DEFAULT FALSE,
    risk_score DECIMAL(3,2) DEFAULT 0.00,
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Comprehensive audit log for all system activities
CREATE TABLE security.audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Actor information
    user_id UUID REFERENCES core.users(id),
    organization_id UUID REFERENCES core.organizations(id),
    session_id UUID REFERENCES security.user_sessions(id),
    api_key_id UUID REFERENCES security.api_keys(id),
    
    -- Action details
    action VARCHAR(100) NOT NULL, -- create, read, update, delete, login, logout, etc.
    resource_type VARCHAR(100) NOT NULL, -- user, company, valuation, etc.
    resource_id UUID,
    
    -- Request context
    endpoint VARCHAR(500),
    http_method VARCHAR(10),
    ip_address INET,
    user_agent TEXT,
    
    -- Change tracking
    old_values JSONB,
    new_values JSONB,
    
    -- Result and metadata
    status VARCHAR(20) DEFAULT 'success' CHECK (status IN ('success', 'failure', 'error')),
    error_message TEXT,
    duration_ms INTEGER,
    
    -- Additional context
    metadata JSONB DEFAULT '{}',
    tags TEXT[],
    
    -- Timestamp
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Data access permissions for fine-grained access control
CREATE TABLE security.data_permissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Subject (who gets permission)
    subject_type VARCHAR(20) NOT NULL CHECK (subject_type IN ('user', 'role', 'organization')),
    subject_id UUID NOT NULL,
    
    -- Object (what is being accessed)
    resource_type VARCHAR(100) NOT NULL,
    resource_id UUID,
    
    -- Permission details
    permission_type VARCHAR(20) NOT NULL CHECK (permission_type IN ('read', 'write', 'delete', 'admin', 'share')),
    granted_by UUID REFERENCES core.users(id),
    granted_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Conditions and constraints
    conditions JSONB DEFAULT '{}', -- IP restrictions, time-based, etc.
    expires_at TIMESTAMPTZ,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    revoked_by UUID REFERENCES core.users(id),
    revoked_at TIMESTAMPTZ,
    revocation_reason TEXT,
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Security incidents and threat detection
CREATE TABLE security.security_incidents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Incident classification
    incident_type VARCHAR(50) NOT NULL, -- brute_force, data_breach, unauthorized_access, etc.
    severity VARCHAR(20) DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'resolved', 'false_positive')),
    
    -- Affected entities
    affected_user_id UUID REFERENCES core.users(id),
    affected_organization_id UUID REFERENCES core.organizations(id),
    affected_resources JSONB DEFAULT '[]',
    
    -- Incident details
    description TEXT NOT NULL,
    detection_method VARCHAR(100), -- automated, manual, external_report
    
    -- Source information
    source_ip INET,
    source_country VARCHAR(2),
    user_agent TEXT,
    
    -- Impact assessment
    data_exposed BOOLEAN DEFAULT FALSE,
    records_affected INTEGER DEFAULT 0,
    financial_impact DECIMAL(10,2),
    
    -- Response actions
    actions_taken TEXT[],
    assigned_to UUID REFERENCES core.users(id),
    
    -- Timeline
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    
    -- Additional context
    metadata JSONB DEFAULT '{}',
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID REFERENCES core.users(id)
);

-- Password reset tokens and email verification
CREATE TABLE security.verification_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES core.users(id) ON DELETE CASCADE,
    
    -- Token details
    token_hash VARCHAR(255) NOT NULL UNIQUE,
    token_type VARCHAR(50) NOT NULL CHECK (token_type IN ('email_verification', 'password_reset', 'two_factor_setup', 'account_recovery')),
    
    -- Lifecycle
    expires_at TIMESTAMPTZ NOT NULL,
    used_at TIMESTAMPTZ,
    is_used BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- =====================================================

-- Enable RLS on sensitive tables
ALTER TABLE core.companies ENABLE ROW LEVEL SECURITY;
ALTER TABLE core.valuations ENABLE ROW LEVEL SECURITY;
ALTER TABLE core.financial_statements ENABLE ROW LEVEL SECURITY;
ALTER TABLE security.audit_log ENABLE ROW LEVEL SECURITY;

-- Companies RLS - users can only see companies in their organization
CREATE POLICY company_access_policy ON core.companies
    FOR ALL
    USING (
        organization_id IN (
            SELECT organization_id 
            FROM core.user_organizations 
            WHERE user_id = current_setting('app.current_user_id')::UUID 
            AND status = 'active'
        )
        OR organization_id IS NULL -- Public companies
    );

-- Valuations RLS - users can only see valuations in their organization or shared ones
CREATE POLICY valuation_access_policy ON core.valuations
    FOR ALL
    USING (
        organization_id IN (
            SELECT organization_id 
            FROM core.user_organizations 
            WHERE user_id = current_setting('app.current_user_id')::UUID 
            AND status = 'active'
        )
        OR access_level = 'public'
        OR created_by = current_setting('app.current_user_id')::UUID
    );

-- Financial statements RLS - follows company access
CREATE POLICY financial_statements_access_policy ON core.financial_statements
    FOR ALL
    USING (
        company_id IN (
            SELECT id FROM core.companies
            WHERE organization_id IN (
                SELECT organization_id 
                FROM core.user_organizations 
                WHERE user_id = current_setting('app.current_user_id')::UUID 
                AND status = 'active'
            )
            OR organization_id IS NULL
        )
    );

-- Audit log RLS - users can only see their own actions or organization-level if admin
CREATE POLICY audit_log_access_policy ON security.audit_log
    FOR SELECT
    USING (
        user_id = current_setting('app.current_user_id')::UUID
        OR organization_id IN (
            SELECT organization_id 
            FROM core.user_organizations 
            WHERE user_id = current_setting('app.current_user_id')::UUID 
            AND status = 'active'
            AND role IN ('owner', 'admin')
        )
    );

-- =====================================================
-- SECURITY INDEXES
-- =====================================================

CREATE INDEX idx_api_keys_user_id ON security.api_keys USING btree (user_id);
CREATE INDEX idx_api_keys_org_id ON security.api_keys USING btree (organization_id);
CREATE INDEX idx_api_keys_hash ON security.api_keys USING btree (key_hash);
CREATE INDEX idx_api_keys_active ON security.api_keys USING btree (is_active) WHERE is_active = TRUE;

CREATE INDEX idx_user_sessions_user_id ON security.user_sessions USING btree (user_id);
CREATE INDEX idx_user_sessions_token ON security.user_sessions USING btree (session_token_hash);
CREATE INDEX idx_user_sessions_active ON security.user_sessions USING btree (is_active) WHERE is_active = TRUE;
CREATE INDEX idx_user_sessions_expires ON security.user_sessions USING btree (expires_at);

-- Audit log indexes for performance
CREATE INDEX idx_audit_log_user_id ON security.audit_log USING btree (user_id);
CREATE INDEX idx_audit_log_timestamp ON security.audit_log USING btree (timestamp DESC);
CREATE INDEX idx_audit_log_resource ON security.audit_log USING btree (resource_type, resource_id);
CREATE INDEX idx_audit_log_action ON security.audit_log USING btree (action);
CREATE INDEX idx_audit_log_ip ON security.audit_log USING btree (ip_address);

-- Partitioning for audit log (by month)
CREATE TABLE security.audit_log_y2024m01 PARTITION OF security.audit_log
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
-- Add more partitions as needed...

CREATE INDEX idx_data_permissions_subject ON security.data_permissions USING btree (subject_type, subject_id);
CREATE INDEX idx_data_permissions_resource ON security.data_permissions USING btree (resource_type, resource_id);
CREATE INDEX idx_data_permissions_active ON security.data_permissions USING btree (is_active) WHERE is_active = TRUE;

CREATE INDEX idx_security_incidents_type ON security.security_incidents USING btree (incident_type);
CREATE INDEX idx_security_incidents_severity ON security.security_incidents USING btree (severity);
CREATE INDEX idx_security_incidents_status ON security.security_incidents USING btree (status);
CREATE INDEX idx_security_incidents_detected ON security.security_incidents USING btree (detected_at DESC);

CREATE INDEX idx_verification_tokens_user ON security.verification_tokens USING btree (user_id);
CREATE INDEX idx_verification_tokens_hash ON security.verification_tokens USING btree (token_hash);
CREATE INDEX idx_verification_tokens_type ON security.verification_tokens USING btree (token_type);
CREATE INDEX idx_verification_tokens_expires ON security.verification_tokens USING btree (expires_at);

-- =====================================================
-- SECURITY FUNCTIONS
-- =====================================================

-- Function to check if user has permission
CREATE OR REPLACE FUNCTION security.has_permission(
    p_user_id UUID,
    p_resource_type VARCHAR,
    p_resource_id UUID,
    p_permission_type VARCHAR
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    has_perm BOOLEAN := FALSE;
BEGIN
    -- Check direct user permissions
    SELECT EXISTS(
        SELECT 1 FROM security.data_permissions dp
        WHERE dp.subject_type = 'user'
        AND dp.subject_id = p_user_id
        AND dp.resource_type = p_resource_type
        AND (dp.resource_id = p_resource_id OR dp.resource_id IS NULL)
        AND dp.permission_type = p_permission_type
        AND dp.is_active = TRUE
        AND (dp.expires_at IS NULL OR dp.expires_at > NOW())
    ) INTO has_perm;
    
    IF has_perm THEN
        RETURN TRUE;
    END IF;
    
    -- Check organization-level permissions
    SELECT EXISTS(
        SELECT 1 FROM security.data_permissions dp
        JOIN core.user_organizations uo ON uo.organization_id = dp.subject_id
        WHERE dp.subject_type = 'organization'
        AND uo.user_id = p_user_id
        AND uo.status = 'active'
        AND dp.resource_type = p_resource_type
        AND (dp.resource_id = p_resource_id OR dp.resource_id IS NULL)
        AND dp.permission_type = p_permission_type
        AND dp.is_active = TRUE
        AND (dp.expires_at IS NULL OR dp.expires_at > NOW())
    ) INTO has_perm;
    
    RETURN has_perm;
END;
$$;

-- Function to log security events
CREATE OR REPLACE FUNCTION security.log_security_event(
    p_incident_type VARCHAR,
    p_severity VARCHAR DEFAULT 'medium',
    p_description TEXT DEFAULT '',
    p_user_id UUID DEFAULT NULL,
    p_source_ip INET DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    incident_id UUID;
BEGIN
    INSERT INTO security.security_incidents (
        incident_type,
        severity,
        description,
        affected_user_id,
        source_ip,
        metadata
    ) VALUES (
        p_incident_type,
        p_severity,
        p_description,
        p_user_id,
        p_source_ip,
        p_metadata
    ) RETURNING id INTO incident_id;
    
    RETURN incident_id;
END;
$$;

-- =====================================================
-- SECURITY TRIGGERS
-- =====================================================

-- Trigger to automatically log sensitive operations
CREATE OR REPLACE FUNCTION security.audit_trigger_function()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    old_data JSONB;
    new_data JSONB;
    action_type VARCHAR(10);
BEGIN
    -- Determine action type
    IF TG_OP = 'DELETE' THEN
        action_type := 'delete';
        old_data := to_jsonb(OLD);
        new_data := '{}';
    ELSIF TG_OP = 'UPDATE' THEN
        action_type := 'update';
        old_data := to_jsonb(OLD);
        new_data := to_jsonb(NEW);
    ELSIF TG_OP = 'INSERT' THEN
        action_type := 'create';
        old_data := '{}';
        new_data := to_jsonb(NEW);
    END IF;
    
    -- Insert audit record
    INSERT INTO security.audit_log (
        action,
        resource_type,
        resource_id,
        old_values,
        new_values,
        user_id
    ) VALUES (
        action_type,
        TG_TABLE_NAME,
        COALESCE(NEW.id, OLD.id),
        old_data,
        new_data,
        NULLIF(current_setting('app.current_user_id', TRUE), '')::UUID
    );
    
    RETURN COALESCE(NEW, OLD);
END;
$$;

-- Apply audit triggers to sensitive tables
CREATE TRIGGER audit_companies_changes
    AFTER INSERT OR UPDATE OR DELETE ON core.companies
    FOR EACH ROW EXECUTE FUNCTION security.audit_trigger_function();

CREATE TRIGGER audit_valuations_changes
    AFTER INSERT OR UPDATE OR DELETE ON core.valuations
    FOR EACH ROW EXECUTE FUNCTION security.audit_trigger_function();

CREATE TRIGGER audit_users_changes
    AFTER INSERT OR UPDATE OR DELETE ON core.users
    FOR EACH ROW EXECUTE FUNCTION security.audit_trigger_function();

-- =====================================================
-- COMMENTS
-- =====================================================

COMMENT ON SCHEMA security IS 'Security, authentication, and audit trail tables';
COMMENT ON TABLE security.api_keys IS 'API keys for external system access with rate limiting';
COMMENT ON TABLE security.user_sessions IS 'Active user sessions with device tracking';
COMMENT ON TABLE security.audit_log IS 'Comprehensive audit trail for all system activities';
COMMENT ON TABLE security.data_permissions IS 'Fine-grained access control permissions';
COMMENT ON TABLE security.security_incidents IS 'Security incident tracking and management';
COMMENT ON TABLE security.verification_tokens IS 'Tokens for email verification and password reset';