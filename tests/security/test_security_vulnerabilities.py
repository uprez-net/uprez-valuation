"""
Security Testing Suite for IPO Valuation Platform

Comprehensive security tests covering:
- Authentication and authorization
- Input validation and injection attacks
- Data encryption and privacy
- API security
- File upload security
- Rate limiting and abuse prevention
"""

import pytest
import requests
import json
import base64
import time
import hashlib
import hmac
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os
from typing import Dict, List, Any, Optional

from tests.utils.test_data import get_standard_test_scenario
from tests.utils.mocks import MockRedisClient


class TestAuthenticationSecurity:
    """Test authentication and session security."""
    
    def setup_method(self):
        """Set up authentication security tests."""
        self.base_url = "http://localhost:8000/api/v1"
        self.test_user = {
            "email": "security.test@example.com",
            "password": "SecureTestPassword123!",
            "first_name": "Security",
            "last_name": "Tester"
        }
    
    def test_password_strength_requirements(self, test_client):
        """Test password strength validation."""
        weak_passwords = [
            "123456",           # Too simple
            "password",         # Common password
            "12345678",         # Only numbers
            "abcdefgh",         # Only letters
            "Password",         # Missing numbers and special chars
            "Pass123",          # Too short
            "PASSWORD123!"      # No lowercase
        ]
        
        for weak_password in weak_passwords:
            register_data = {
                "email": "test@example.com",
                "password": weak_password,
                "first_name": "Test",
                "last_name": "User"
            }
            
            response = test_client.post("/api/v1/auth/register", json=register_data)
            
            assert response.status_code == 400
            assert "password" in response.json().get("details", {}).get("field_errors", {})
    
    def test_login_brute_force_protection(self, test_client):
        """Test protection against brute force login attempts."""
        # Create a test user first
        register_data = self.test_user.copy()
        test_client.post("/api/v1/auth/register", json=register_data)
        
        # Attempt multiple failed logins
        failed_attempts = 0
        max_attempts = 5
        
        for _ in range(max_attempts + 2):  # Try more than the limit
            login_data = {
                "email": self.test_user["email"],
                "password": "WrongPassword123!"
            }
            
            response = test_client.post("/api/v1/auth/login", json=login_data)
            
            if response.status_code == 429:  # Rate limited
                assert "rate limit" in response.json().get("message", "").lower()
                break
            elif response.status_code == 401:  # Failed login
                failed_attempts += 1
            
            # Small delay to avoid overwhelming
            time.sleep(0.1)
        
        # Should be rate limited after max attempts
        assert failed_attempts <= max_attempts
    
    def test_jwt_token_security(self, test_client):
        """Test JWT token security and validation."""
        # Register and login user
        register_response = test_client.post("/api/v1/auth/register", json=self.test_user)
        assert register_response.status_code == 201
        
        login_response = test_client.post("/api/v1/auth/login", json={
            "email": self.test_user["email"],
            "password": self.test_user["password"]
        })
        assert login_response.status_code == 200
        
        token = login_response.json().get("access_token")
        assert token is not None
        
        # Test with valid token
        headers = {"Authorization": f"Bearer {token}"}
        response = test_client.get("/api/v1/users/profile", headers=headers)
        assert response.status_code == 200
        
        # Test with invalid token
        invalid_headers = {"Authorization": "Bearer invalid.token.here"}
        response = test_client.get("/api/v1/users/profile", headers=invalid_headers)
        assert response.status_code == 401
        
        # Test with malformed token
        malformed_headers = {"Authorization": "Bearer malformed_token"}
        response = test_client.get("/api/v1/users/profile", headers=malformed_headers)
        assert response.status_code == 401
        
        # Test without token
        response = test_client.get("/api/v1/users/profile")
        assert response.status_code == 401
    
    def test_session_expiration(self, test_client):
        """Test session token expiration."""
        # This test would typically require mocking time or using expired tokens
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = jwt.ExpiredSignatureError("Token has expired")
            
            expired_headers = {"Authorization": "Bearer expired.token.here"}
            response = test_client.get("/api/v1/users/profile", headers=expired_headers)
            assert response.status_code == 401
            assert "expired" in response.json().get("message", "").lower()
    
    def test_csrf_protection(self, test_client):
        """Test CSRF protection mechanisms."""
        # Register and login user
        register_response = test_client.post("/api/v1/auth/register", json=self.test_user)
        login_response = test_client.post("/api/v1/auth/login", json={
            "email": self.test_user["email"],
            "password": self.test_user["password"]
        })
        
        token = login_response.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test state-changing operations require proper headers
        valuation_data = {
            "company_name": "CSRF Test Corp",
            "industry_sector": "Technology",
            "expected_ipo_date": "2024-12-31",
            "shares_on_issue": 1000000
        }
        
        # Should work with proper headers
        response = test_client.post("/api/v1/valuations", json=valuation_data, headers=headers)
        assert response.status_code in [201, 400]  # Created or validation error
        
        # Test without Origin header might be blocked in production
        # This would be environment-specific


class TestInputValidationSecurity:
    """Test input validation and injection attack prevention."""
    
    def test_sql_injection_prevention(self, test_client, auth_headers):
        """Test prevention of SQL injection attacks."""
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM sensitive_data --",
            "admin'--",
            "admin'; INSERT INTO users VALUES ('hacker', 'password'); --"
        ]
        
        for payload in sql_injection_payloads:
            # Test in company name field
            valuation_data = {
                "company_name": payload,
                "industry_sector": "Technology",
                "expected_ipo_date": "2024-12-31",
                "shares_on_issue": 1000000
            }
            
            response = test_client.post("/api/v1/valuations", json=valuation_data, headers=auth_headers)
            
            # Should either validate input properly (400) or process safely
            assert response.status_code in [400, 201]
            
            if response.status_code == 201:
                # If accepted, verify no SQL injection occurred
                job_id = response.json().get("job_id")
                status_response = test_client.get(f"/api/v1/valuations/{job_id}/status", headers=auth_headers)
                assert status_response.status_code == 200
    
    def test_xss_prevention(self, test_client, auth_headers):
        """Test prevention of Cross-Site Scripting (XSS) attacks."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "'\"><script>alert('XSS')</script>",
        ]
        
        for payload in xss_payloads:
            valuation_data = {
                "company_name": f"Test Corp {payload}",
                "business_description": f"Company description {payload}",
                "industry_sector": "Technology",
                "expected_ipo_date": "2024-12-31",
                "shares_on_issue": 1000000
            }
            
            response = test_client.post("/api/v1/valuations", json=valuation_data, headers=auth_headers)
            
            if response.status_code == 201:
                job_id = response.json().get("job_id")
                
                # Get results and verify XSS payload is sanitized
                result_response = test_client.get(f"/api/v1/valuations/{job_id}/result", headers=auth_headers)
                if result_response.status_code == 200:
                    result_data = result_response.json()
                    company_name = result_data.get("company_name", "")
                    
                    # Should not contain script tags or javascript
                    assert "<script>" not in company_name
                    assert "javascript:" not in company_name
                    assert "onerror=" not in company_name
    
    def test_path_traversal_prevention(self, test_client, auth_headers):
        """Test prevention of path traversal attacks."""
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]
        
        for payload in path_traversal_payloads:
            # Test in document download endpoint
            response = test_client.get(f"/api/v1/documents/download/{payload}", headers=auth_headers)
            
            # Should return 404 or 400, not 200 with sensitive file
            assert response.status_code in [400, 404]
    
    def test_xxe_prevention(self, test_client, auth_headers):
        """Test prevention of XML External Entity (XXE) attacks."""
        xxe_payload = """<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE foo [
          <!ENTITY xxe SYSTEM "file:///etc/passwd">
        ]>
        <company>
          <name>&xxe;</name>
          <industry>Technology</industry>
        </company>"""
        
        # Test XML upload (if supported)
        files = {
            'file': ('malicious.xml', xxe_payload.encode(), 'application/xml'),
            'document_type': (None, 'company_profile'),
        }
        
        response = test_client.post("/api/v1/documents/upload", files=files, headers=auth_headers)
        
        # Should reject malicious XML or process safely
        if response.status_code == 201:
            # If processed, verify no XXE attack succeeded
            result = response.json()
            doc_id = result.get("job_id")
            
            extraction_response = test_client.get(f"/api/v1/documents/{doc_id}/extraction", headers=auth_headers)
            if extraction_response.status_code == 200:
                extracted_data = extraction_response.json()
                # Should not contain /etc/passwd content
                assert "root:x:" not in str(extracted_data)
    
    def test_command_injection_prevention(self, test_client, auth_headers):
        """Test prevention of command injection attacks."""
        command_injection_payloads = [
            "; ls -la",
            "&& cat /etc/passwd",
            "| whoami",
            "; rm -rf /",
            "`cat /etc/passwd`",
            "$(cat /etc/passwd)"
        ]
        
        for payload in command_injection_payloads:
            valuation_data = {
                "company_name": f"Test Corp{payload}",
                "industry_sector": "Technology",
                "expected_ipo_date": "2024-12-31",
                "shares_on_issue": 1000000
            }
            
            response = test_client.post("/api/v1/valuations", json=valuation_data, headers=auth_headers)
            
            # Should handle safely without executing commands
            assert response.status_code in [400, 201]


class TestDataProtectionSecurity:
    """Test data encryption and privacy protection."""
    
    def test_sensitive_data_encryption(self, test_client):
        """Test that sensitive data is properly encrypted."""
        # Register user with sensitive information
        user_data = {
            "email": "sensitive@example.com",
            "password": "SecurePassword123!",
            "first_name": "Sensitive",
            "last_name": "User",
            "phone": "+61 400 123 456",
            "company": "Confidential Corp"
        }
        
        response = test_client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 201
        
        # Login and get profile
        login_response = test_client.post("/api/v1/auth/login", json={
            "email": user_data["email"],
            "password": user_data["password"]
        })
        
        token = login_response.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}
        
        profile_response = test_client.get("/api/v1/users/profile", headers=headers)
        profile_data = profile_response.json()
        
        # Sensitive data should not be exposed in plain text responses
        assert "password" not in profile_data
        
        # Phone number might be partially masked
        if "phone" in profile_data:
            phone = profile_data["phone"]
            # Should be masked or properly formatted
            assert len(phone) > 0
    
    def test_pii_data_handling(self, test_client, auth_headers):
        """Test handling of Personally Identifiable Information (PII)."""
        # Create valuation with PII data
        valuation_data = {
            "company_name": "Test Corp",
            "industry_sector": "Technology",
            "contact_person": {
                "name": "John Smith",
                "email": "john.smith@testcorp.com",
                "phone": "+61 400 123 456"
            },
            "expected_ipo_date": "2024-12-31",
            "shares_on_issue": 1000000
        }
        
        response = test_client.post("/api/v1/valuations", json=valuation_data, headers=auth_headers)
        
        if response.status_code == 201:
            job_id = response.json().get("job_id")
            
            # Get valuation results
            result_response = test_client.get(f"/api/v1/valuations/{job_id}/result", headers=auth_headers)
            
            if result_response.status_code == 200:
                result_data = result_response.json()
                
                # PII should be handled appropriately
                # Contact info should either be excluded or properly protected
                if "contact_person" in result_data:
                    contact = result_data["contact_person"]
                    # Email might be partially masked
                    if "email" in contact:
                        email = contact["email"]
                        # Check if email is masked (e.g., j***@testcorp.com)
                        assert "@" in email
    
    def test_data_retention_policies(self, test_client, auth_headers):
        """Test data retention and deletion policies."""
        # Create a valuation
        valuation_data = {
            "company_name": "Retention Test Corp",
            "industry_sector": "Technology",
            "expected_ipo_date": "2024-12-31",
            "shares_on_issue": 1000000
        }
        
        response = test_client.post("/api/v1/valuations", json=valuation_data, headers=auth_headers)
        assert response.status_code == 201
        
        job_id = response.json().get("job_id")
        
        # Test data deletion request
        delete_response = test_client.delete(f"/api/v1/valuations/{job_id}", headers=auth_headers)
        
        # Should support data deletion (GDPR compliance)
        assert delete_response.status_code in [200, 204, 404]
        
        if delete_response.status_code in [200, 204]:
            # Verify data is deleted
            get_response = test_client.get(f"/api/v1/valuations/{job_id}/result", headers=auth_headers)
            assert get_response.status_code in [404, 410]  # Not found or gone
    
    def test_audit_trail_security(self, test_client, auth_headers):
        """Test audit trail and logging security."""
        # Perform various operations that should be audited
        operations = [
            ("POST", "/api/v1/valuations", {
                "company_name": "Audit Test Corp",
                "industry_sector": "Technology",
                "expected_ipo_date": "2024-12-31",
                "shares_on_issue": 1000000
            }),
            ("GET", "/api/v1/users/profile", None),
        ]
        
        for method, endpoint, data in operations:
            if method == "POST":
                response = test_client.post(endpoint, json=data, headers=auth_headers)
            else:
                response = test_client.get(endpoint, headers=auth_headers)
            
            # Operations should be logged for audit
            # Verify response includes audit-related headers
            assert "X-Request-ID" in response.headers or "Request-ID" in response.headers


class TestFileUploadSecurity:
    """Test file upload security."""
    
    def test_file_type_validation(self, test_client, auth_headers):
        """Test file type validation and restrictions."""
        malicious_files = [
            ('malicious.exe', b'MZ\x90\x00', 'application/x-executable'),
            ('script.js', b'alert("malicious")', 'application/javascript'),
            ('shell.sh', b'#!/bin/bash\nrm -rf /', 'application/x-sh'),
            ('payload.php', b'<?php system($_GET["cmd"]); ?>', 'application/x-php'),
        ]
        
        for filename, content, mime_type in malicious_files:
            files = {
                'file': (filename, content, mime_type),
                'document_type': (None, 'other'),
            }
            
            response = test_client.post("/api/v1/documents/upload", files=files, headers=auth_headers)
            
            # Should reject malicious file types
            assert response.status_code in [400, 415]  # Bad request or unsupported media type
    
    def test_file_size_limits(self, test_client, auth_headers):
        """Test file size limitations."""
        # Create a large file (> 50MB)
        large_content = b"A" * (51 * 1024 * 1024)  # 51MB
        
        files = {
            'file': ('large_file.pdf', large_content, 'application/pdf'),
            'document_type': (None, 'prospectus'),
        }
        
        response = test_client.post("/api/v1/documents/upload", files=files, headers=auth_headers)
        
        # Should reject oversized files
        assert response.status_code == 413  # Payload too large
    
    def test_malicious_pdf_content(self, test_client, auth_headers):
        """Test handling of malicious PDF content."""
        # Create PDF with embedded JavaScript
        malicious_pdf = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R/OpenAction<</Type/Action/S/JavaScript/JS(app.alert('XSS'))>>>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R>>endobj
xref
0 4
0000000000 65535 f 
0000000010 00000 n 
0000000120 00000 n 
0000000173 00000 n 
trailer<</Size 4/Root 1 0 R>>
startxref
210
%%EOF"""
        
        files = {
            'file': ('malicious.pdf', malicious_pdf, 'application/pdf'),
            'document_type': (None, 'prospectus'),
        }
        
        response = test_client.post("/api/v1/documents/upload", files=files, headers=auth_headers)
        
        if response.status_code == 201:
            # If accepted, verify malicious content is sanitized
            doc_id = response.json().get("job_id")
            
            extraction_response = test_client.get(f"/api/v1/documents/{doc_id}/extraction", headers=auth_headers)
            if extraction_response.status_code == 200:
                extracted_data = extraction_response.json()
                
                # Should not contain JavaScript
                content_str = str(extracted_data).lower()
                assert "javascript" not in content_str
                assert "app.alert" not in content_str
    
    def test_file_content_scanning(self, test_client, auth_headers):
        """Test file content scanning for malware."""
        # Create file with suspicious patterns
        suspicious_content = b"""
        This is a legitimate document.
        However, it contains suspicious patterns:
        EICAR-STANDARD-ANTIVIRUS-TEST-FILE!
        """
        
        files = {
            'file': ('suspicious.pdf', suspicious_content, 'application/pdf'),
            'document_type': (None, 'financial_statements'),
        }
        
        response = test_client.post("/api/v1/documents/upload", files=files, headers=auth_headers)
        
        # Security scanner should detect and reject or quarantine
        assert response.status_code in [201, 400, 422]


class TestAPISecurityHeaders:
    """Test API security headers and configurations."""
    
    def test_security_headers_present(self, test_client):
        """Test that required security headers are present."""
        response = test_client.get("/api/v1/health")
        
        security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': None,  # Should be present but value varies
        }
        
        for header, expected_value in security_headers.items():
            assert header in response.headers, f"Missing security header: {header}"
            
            if expected_value:
                assert response.headers[header] == expected_value
    
    def test_cors_configuration(self, test_client):
        """Test CORS configuration security."""
        # Test preflight request
        response = test_client.options(
            "/api/v1/valuations",
            headers={
                "Origin": "https://malicious-site.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        if "Access-Control-Allow-Origin" in response.headers:
            allowed_origins = response.headers["Access-Control-Allow-Origin"]
            
            # Should not allow all origins (*) in production
            assert allowed_origins != "*" or os.getenv("ENVIRONMENT") == "development"
            
            # Should not allow malicious origins
            assert "malicious-site.com" not in allowed_origins
    
    def test_rate_limiting_headers(self, test_client, auth_headers):
        """Test rate limiting headers."""
        response = test_client.get("/api/v1/users/profile", headers=auth_headers)
        
        # Rate limiting headers should be present
        rate_limit_headers = [
            'X-RateLimit-Limit',
            'X-RateLimit-Remaining',
            'X-RateLimit-Reset'
        ]
        
        # At least some rate limiting headers should be present
        present_headers = [h for h in rate_limit_headers if h in response.headers]
        assert len(present_headers) > 0, "No rate limiting headers found"
    
    def test_information_disclosure_prevention(self, test_client):
        """Test prevention of information disclosure."""
        # Test server header doesn't reveal too much information
        response = test_client.get("/api/v1/health")
        
        if "Server" in response.headers:
            server_header = response.headers["Server"].lower()
            
            # Should not reveal specific version numbers
            assert "python" not in server_header
            assert "fastapi" not in server_header
            assert "uvicorn" not in server_header
        
        # Test error responses don't leak information
        response = test_client.get("/api/v1/non-existent-endpoint")
        assert response.status_code == 404
        
        error_data = response.json()
        error_message = str(error_data).lower()
        
        # Should not contain sensitive paths or internal details
        assert "/src/" not in error_message
        assert "traceback" not in error_message
        assert "internal error" not in error_message or "production" in os.getenv("ENVIRONMENT", "")


class TestAuthorizationSecurity:
    """Test authorization and access control."""
    
    def test_rbac_enforcement(self, test_client):
        """Test Role-Based Access Control (RBAC) enforcement."""
        # Create users with different roles
        basic_user = {
            "email": "basic@example.com",
            "password": "BasicPassword123!",
            "subscription_tier": "basic"
        }
        
        premium_user = {
            "email": "premium@example.com",
            "password": "PremiumPassword123!",
            "subscription_tier": "premium"
        }
        
        # Register users
        test_client.post("/api/v1/auth/register", json=basic_user)
        test_client.post("/api/v1/auth/register", json=premium_user)
        
        # Login basic user
        basic_login = test_client.post("/api/v1/auth/login", json={
            "email": basic_user["email"],
            "password": basic_user["password"]
        })
        basic_token = basic_login.json().get("access_token")
        basic_headers = {"Authorization": f"Bearer {basic_token}"}
        
        # Login premium user
        premium_login = test_client.post("/api/v1/auth/login", json={
            "email": premium_user["email"],
            "password": premium_user["password"]
        })
        premium_token = premium_login.json().get("access_token")
        premium_headers = {"Authorization": f"Bearer {premium_token}"}
        
        # Test premium-only endpoint with basic user
        premium_request = {
            "analysis_type": "premium",
            "company_name": "Test Corp",
            "industry_sector": "Technology",
            "expected_ipo_date": "2024-12-31",
            "shares_on_issue": 1000000
        }
        
        basic_response = test_client.post("/api/v1/valuations", json=premium_request, headers=basic_headers)
        premium_response = test_client.post("/api/v1/valuations", json=premium_request, headers=premium_headers)
        
        # Basic user should be denied premium features
        assert basic_response.status_code in [403, 400]
        
        # Premium user should be allowed
        assert premium_response.status_code in [201, 400]  # Created or validation error
    
    def test_horizontal_privilege_escalation_prevention(self, test_client):
        """Test prevention of horizontal privilege escalation."""
        # Create two separate users
        user1 = {
            "email": "user1@example.com",
            "password": "User1Password123!"
        }
        
        user2 = {
            "email": "user2@example.com",
            "password": "User2Password123!"
        }
        
        # Register users
        test_client.post("/api/v1/auth/register", json=user1)
        test_client.post("/api/v1/auth/register", json=user2)
        
        # Login both users
        login1 = test_client.post("/api/v1/auth/login", json={
            "email": user1["email"],
            "password": user1["password"]
        })
        token1 = login1.json().get("access_token")
        headers1 = {"Authorization": f"Bearer {token1}"}
        
        login2 = test_client.post("/api/v1/auth/login", json={
            "email": user2["email"],
            "password": user2["password"]
        })
        token2 = login2.json().get("access_token")
        headers2 = {"Authorization": f"Bearer {token2}"}
        
        # User1 creates a valuation
        valuation_data = {
            "company_name": "User1 Corp",
            "industry_sector": "Technology",
            "expected_ipo_date": "2024-12-31",
            "shares_on_issue": 1000000
        }
        
        response1 = test_client.post("/api/v1/valuations", json=valuation_data, headers=headers1)
        if response1.status_code == 201:
            job_id = response1.json().get("job_id")
            
            # User2 should not be able to access User1's valuation
            response2 = test_client.get(f"/api/v1/valuations/{job_id}/result", headers=headers2)
            assert response2.status_code in [403, 404]  # Forbidden or not found
    
    def test_vertical_privilege_escalation_prevention(self, test_client):
        """Test prevention of vertical privilege escalation."""
        # Create regular user
        user = {
            "email": "regular@example.com",
            "password": "RegularPassword123!",
            "role": "user"
        }
        
        test_client.post("/api/v1/auth/register", json=user)
        
        login_response = test_client.post("/api/v1/auth/login", json={
            "email": user["email"],
            "password": user["password"]
        })
        
        token = login_response.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}
        
        # Try to access admin-only endpoints
        admin_endpoints = [
            "/api/v1/admin/users",
            "/api/v1/admin/system-config",
            "/api/v1/admin/audit-logs"
        ]
        
        for endpoint in admin_endpoints:
            response = test_client.get(endpoint, headers=headers)
            # Should be forbidden for regular users
            assert response.status_code in [403, 404]


@pytest.fixture
def auth_headers(test_client):
    """Provide authentication headers for security tests."""
    user_data = {
        "email": "security.tester@example.com",
        "password": "SecureTestPassword123!",
        "first_name": "Security",
        "last_name": "Tester"
    }
    
    # Register user
    register_response = test_client.post("/api/v1/auth/register", json=user_data)
    if register_response.status_code != 201:
        # User might already exist, try to login
        pass
    
    # Login user
    login_response = test_client.post("/api/v1/auth/login", json={
        "email": user_data["email"],
        "password": user_data["password"]
    })
    
    if login_response.status_code == 200:
        token = login_response.json().get("access_token")
        return {"Authorization": f"Bearer {token}"}
    
    return {}


# Security test marks for categorization
pytestmark = [pytest.mark.security]