"""
API documentation generation and client SDK utilities
"""

from typing import Dict, Any, List
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import json


def generate_custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Generate custom OpenAPI schema with additional metadata"""
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://uprez.com/logo.png"
    }
    
    # Add authentication information
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        },
        "apiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    # Add examples for complex schemas
    openapi_schema["components"]["examples"] = {
        "CompanyExample": {
            "summary": "Example company",
            "value": {
                "name": "Commonwealth Bank of Australia",
                "asx_code": "CBA",
                "industry": "financial_services",
                "market_cap": 150000000000
            }
        },
        "ValuationExample": {
            "summary": "Example valuation request",
            "value": {
                "company_id": 1,
                "model_id": 1,
                "name": "CBA DCF Valuation",
                "assumptions": {
                    "discount_rate": 0.10,
                    "terminal_growth_rate": 0.025,
                    "forecast_years": 5
                }
            }
        }
    }
    
    # Add tags with descriptions
    openapi_schema["tags"] = [
        {
            "name": "Authentication",
            "description": "User authentication and authorization"
        },
        {
            "name": "Users",
            "description": "User management and profiles"
        },
        {
            "name": "Companies",
            "description": "Company data management"
        },
        {
            "name": "Valuations",
            "description": "Valuation models and calculations"
        },
        {
            "name": "Documents",
            "description": "Document upload and processing"
        },
        {
            "name": "Collaboration",
            "description": "Project collaboration features"
        },
        {
            "name": "Integrations",
            "description": "External data integrations"
        },
        {
            "name": "Administration",
            "description": "System administration"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


class ClientSDKGenerator:
    """Generate client SDKs for different languages"""
    
    @staticmethod
    def generate_python_client() -> str:
        """Generate Python client SDK"""
        return """
# IPO Valuation Platform Python SDK
import requests
from typing import Dict, Any, Optional, List


class UprezClient:
    def __init__(self, api_key: str, base_url: str = "https://api.uprez.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        })
    
    # Authentication
    def login(self, email: str, password: str) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"email": email, "password": password}
        )
        return response.json()
    
    # Companies
    def list_companies(self, **filters) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/api/v1/companies",
            params=filters
        )
        return response.json()
    
    def get_company(self, company_id: int) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/api/v1/companies/{company_id}"
        )
        return response.json()
    
    def create_company(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/api/v1/companies",
            json=company_data
        )
        return response.json()
    
    # Valuations
    def calculate_valuation(
        self, 
        company_id: int, 
        model_id: int, 
        assumptions: Dict[str, Any]
    ) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/api/v1/valuations/calculate",
            json={
                "company_id": company_id,
                "model_id": model_id,
                "assumptions": assumptions
            }
        )
        return response.json()
    
    def get_valuation_result(self, result_id: int) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/api/v1/valuations/results/{result_id}"
        )
        return response.json()


# Usage example:
# client = UprezClient("your-api-key")
# companies = client.list_companies(industry="technology")
# valuation = client.calculate_valuation(1, 1, {"discount_rate": 0.10})
"""
    
    @staticmethod
    def generate_javascript_client() -> str:
        """Generate JavaScript/TypeScript client SDK"""
        return """
// IPO Valuation Platform JavaScript SDK
class UprezClient {
    constructor(apiKey, baseUrl = 'https://api.uprez.com') {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
        this.headers = {
            'X-API-Key': apiKey,
            'Content-Type': 'application/json'
        };
    }

    async request(method, endpoint, data = null) {
        const url = `${this.baseUrl}${endpoint}`;
        const options = {
            method,
            headers: this.headers
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(url, options);
        return response.json();
    }

    // Authentication
    async login(email, password) {
        return this.request('POST', '/api/v1/auth/login', { email, password });
    }

    // Companies
    async listCompanies(filters = {}) {
        const params = new URLSearchParams(filters);
        return this.request('GET', `/api/v1/companies?${params}`);
    }

    async getCompany(companyId) {
        return this.request('GET', `/api/v1/companies/${companyId}`);
    }

    async createCompany(companyData) {
        return this.request('POST', '/api/v1/companies', companyData);
    }

    // Valuations
    async calculateValuation(companyId, modelId, assumptions) {
        return this.request('POST', '/api/v1/valuations/calculate', {
            company_id: companyId,
            model_id: modelId,
            assumptions
        });
    }

    async getValuationResult(resultId) {
        return this.request('GET', `/api/v1/valuations/results/${resultId}`);
    }
}

// Usage example:
// const client = new UprezClient('your-api-key');
// const companies = await client.listCompanies({ industry: 'technology' });
// const valuation = await client.calculateValuation(1, 1, { discount_rate: 0.10 });

export default UprezClient;
"""


def generate_postman_collection(app: FastAPI) -> Dict[str, Any]:
    """Generate Postman collection for API testing"""
    
    collection = {
        "info": {
            "name": "IPO Valuation Platform API",
            "description": "Complete API collection for the IPO Valuation Platform",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "auth": {
            "type": "bearer",
            "bearer": [
                {
                    "key": "token",
                    "value": "{{access_token}}",
                    "type": "string"
                }
            ]
        },
        "variable": [
            {
                "key": "baseUrl",
                "value": "http://localhost:8000",
                "type": "string"
            },
            {
                "key": "access_token",
                "value": "",
                "type": "string"
            }
        ],
        "item": [
            {
                "name": "Authentication",
                "item": [
                    {
                        "name": "Login",
                        "request": {
                            "method": "POST",
                            "header": [],
                            "body": {
                                "mode": "raw",
                                "raw": json.dumps({
                                    "email": "user@example.com",
                                    "password": "password123"
                                }),
                                "options": {
                                    "raw": {
                                        "language": "json"
                                    }
                                }
                            },
                            "url": {
                                "raw": "{{baseUrl}}/api/v1/auth/login",
                                "host": ["{{baseUrl}}"],
                                "path": ["api", "v1", "auth", "login"]
                            }
                        }
                    }
                ]
            },
            {
                "name": "Companies",
                "item": [
                    {
                        "name": "List Companies",
                        "request": {
                            "method": "GET",
                            "url": {
                                "raw": "{{baseUrl}}/api/v1/companies",
                                "host": ["{{baseUrl}}"],
                                "path": ["api", "v1", "companies"]
                            }
                        }
                    },
                    {
                        "name": "Create Company",
                        "request": {
                            "method": "POST",
                            "body": {
                                "mode": "raw",
                                "raw": json.dumps({
                                    "name": "Example Company",
                                    "asx_code": "EXC",
                                    "industry": "technology"
                                }),
                                "options": {
                                    "raw": {
                                        "language": "json"
                                    }
                                }
                            },
                            "url": {
                                "raw": "{{baseUrl}}/api/v1/companies",
                                "host": ["{{baseUrl}}"],
                                "path": ["api", "v1", "companies"]
                            }
                        }
                    }
                ]
            }
        ]
    }
    
    return collection