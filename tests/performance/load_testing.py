"""
Load Testing Suite for IPO Valuation Platform

Comprehensive load testing using Locust to simulate realistic user behavior
and validate system performance under various load conditions.
"""

import random
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser
import logging
import csv
import os

# Configure logging for load tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IPOValuationUser(HttpUser):
    """
    Simulates a typical user of the IPO Valuation Platform.
    
    User behavior includes:
    - Authentication
    - Creating valuation requests
    - Monitoring valuation progress
    - Viewing results
    - Downloading reports
    """
    
    wait_time = between(5, 15)  # Wait 5-15 seconds between tasks
    
    def on_start(self):
        """Initialize user session - login and setup."""
        self.auth_token = None
        self.user_data = self._generate_user_data()
        self.valuation_jobs = []  # Track active valuation jobs
        
        # Authenticate user
        self.login()
    
    def on_stop(self):
        """Cleanup when user session ends."""
        if self.auth_token:
            self.logout()
    
    def login(self):
        """Authenticate user and obtain auth token."""
        login_data = {
            "email": self.user_data["email"],
            "password": self.user_data["password"]
        }
        
        with self.client.post(
            "/api/v1/auth/login", 
            json=login_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                self.auth_token = result.get("access_token")
                response.success()
                logger.info(f"User {self.user_data['email']} logged in successfully")
            else:
                response.failure(f"Login failed with status {response.status_code}")
    
    def logout(self):
        """Logout user."""
        if self.auth_token:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            with self.client.post("/api/v1/auth/logout", headers=headers):
                pass
    
    @property
    def auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        if self.auth_token:
            return {"Authorization": f"Bearer {self.auth_token}"}
        return {}
    
    @task(10)
    def create_valuation_request(self):
        """Create a new valuation request - most common user action."""
        if not self.auth_token:
            return
        
        company_data = self._generate_company_data()
        
        with self.client.post(
            "/api/v1/valuations",
            json=company_data,
            headers=self.auth_headers,
            catch_response=True
        ) as response:
            if response.status_code == 201:
                result = response.json()
                job_id = result.get("job_id")
                if job_id:
                    self.valuation_jobs.append({
                        "job_id": job_id,
                        "created_at": time.time(),
                        "status": "queued"
                    })
                    logger.info(f"Created valuation job: {job_id}")
                response.success()
            else:
                response.failure(f"Valuation creation failed: {response.status_code}")
    
    @task(8)
    def check_valuation_status(self):
        """Check status of existing valuation jobs."""
        if not self.auth_token or not self.valuation_jobs:
            return
        
        # Check a random active job
        job = random.choice(self.valuation_jobs)
        job_id = job["job_id"]
        
        with self.client.get(
            f"/api/v1/valuations/{job_id}/status",
            headers=self.auth_headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                job["status"] = result.get("status", "unknown")
                response.success()
            else:
                response.failure(f"Status check failed: {response.status_code}")
    
    @task(5)
    def view_valuation_results(self):
        """View completed valuation results."""
        if not self.auth_token:
            return
        
        # Look for completed jobs
        completed_jobs = [j for j in self.valuation_jobs if j.get("status") == "completed"]
        if not completed_jobs:
            return
        
        job = random.choice(completed_jobs)
        job_id = job["job_id"]
        
        with self.client.get(
            f"/api/v1/valuations/{job_id}/result",
            headers=self.auth_headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
                logger.info(f"Viewed results for job: {job_id}")
            else:
                response.failure(f"Result view failed: {response.status_code}")
    
    @task(3)
    def download_report(self):
        """Download PDF report for completed valuation."""
        if not self.auth_token:
            return
        
        completed_jobs = [j for j in self.valuation_jobs if j.get("status") == "completed"]
        if not completed_jobs:
            return
        
        job = random.choice(completed_jobs)
        job_id = job["job_id"]
        
        with self.client.get(
            f"/api/v1/reports/{job_id}",
            headers=self.auth_headers,
            catch_response=True,
            stream=True
        ) as response:
            if response.status_code == 200:
                # Simulate downloading large PDF
                content_length = int(response.headers.get('content-length', 0))
                response.success()
                logger.info(f"Downloaded report for job: {job_id} ({content_length} bytes)")
            else:
                response.failure(f"Report download failed: {response.status_code}")
    
    @task(2)
    def upload_document(self):
        """Upload a document for analysis."""
        if not self.auth_token:
            return
        
        # Simulate file upload
        files = {
            'file': ('test-prospectus.pdf', self._generate_mock_pdf_content(), 'application/pdf'),
            'document_type': (None, 'prospectus'),
        }
        
        with self.client.post(
            "/api/v1/documents/upload",
            files=files,
            headers=self.auth_headers,
            catch_response=True
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(f"Document upload failed: {response.status_code}")
    
    @task(4)
    def get_market_data(self):
        """Fetch market data for peer analysis."""
        if not self.auth_token:
            return
        
        tickers = ['CBA', 'WBC', 'ANZ', 'BHP', 'CSL', 'WES']
        ticker = random.choice(tickers)
        
        with self.client.get(
            f"/api/v1/market-data/companies/{ticker}",
            headers=self.auth_headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Market data fetch failed: {response.status_code}")
    
    @task(3)
    def search_peer_companies(self):
        """Search for peer companies."""
        if not self.auth_token:
            return
        
        search_criteria = {
            "industry_sector": random.choice(["Technology", "Healthcare", "Financial Services"]),
            "market_cap_range": {
                "min": 100000000,
                "max": 5000000000
            },
            "max_peers": 10
        }
        
        with self.client.post(
            "/api/v1/market-data/peer-groups",
            json=search_criteria,
            headers=self.auth_headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Peer search failed: {response.status_code}")
    
    @task(2)
    def get_user_profile(self):
        """Get user profile information."""
        if not self.auth_token:
            return
        
        with self.client.get(
            "/api/v1/users/profile",
            headers=self.auth_headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Profile fetch failed: {response.status_code}")
    
    @task(1)
    def get_industry_benchmarks(self):
        """Get industry benchmark data."""
        if not self.auth_token:
            return
        
        industry = random.choice(["Technology", "Healthcare", "Financial Services"])
        
        with self.client.get(
            f"/api/v1/analytics/industry-benchmarks?industry_sector={industry}",
            headers=self.auth_headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Benchmarks fetch failed: {response.status_code}")
    
    def _generate_user_data(self) -> Dict[str, str]:
        """Generate realistic user data for testing."""
        user_id = random.randint(1000, 9999)
        return {
            "email": f"test.user.{user_id}@example.com",
            "password": "TestPassword123!",
            "first_name": f"TestUser{user_id}",
            "last_name": "Smith",
            "company": f"Test Company {user_id}"
        }
    
    def _generate_company_data(self) -> Dict[str, Any]:
        """Generate realistic company data for valuation requests."""
        company_id = random.randint(10000, 99999)
        
        industries = [
            "Technology", "Healthcare", "Financial Services", 
            "Materials", "Energy", "Consumer Discretionary"
        ]
        
        # Generate IPO date 30-180 days in the future
        ipo_date = datetime.now() + timedelta(days=random.randint(30, 180))
        
        return {
            "company_name": f"Test Corp {company_id} Pty Ltd",
            "abn": f"{random.randint(10000000000, 99999999999)}",
            "industry_sector": random.choice(industries),
            "business_description": f"A leading company in the {random.choice(industries).lower()} sector with innovative solutions and strong market position.",
            "expected_ipo_date": ipo_date.strftime("%Y-%m-%d"),
            "expected_offer_price_range": {
                "min": round(random.uniform(1.0, 4.0), 2),
                "max": round(random.uniform(4.0, 10.0), 2)
            },
            "shares_on_issue": random.randint(10000000, 100000000),
            "analysis_type": random.choice(["basic", "comprehensive", "premium"]),
            "include_peer_analysis": random.choice([True, False]),
            "risk_assessment_level": random.choice(["basic", "detailed", "comprehensive"])
        }
    
    def _generate_mock_pdf_content(self) -> bytes:
        """Generate mock PDF content for file uploads."""
        # Simple mock PDF header
        pdf_content = b"%PDF-1.4\n"
        pdf_content += b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        pdf_content += b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        pdf_content += b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Contents 4 0 R >>\nendobj\n"
        pdf_content += b"4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test Document) Tj\nET\nendstream\nendobj\n"
        pdf_content += b"xref\n0 5\n0000000000 65535 f\n"
        pdf_content += b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n%%EOF"
        return pdf_content


class PremiumUser(IPOValuationUser):
    """Premium user with higher usage patterns and access to advanced features."""
    
    weight = 3  # Premium users are less common
    wait_time = between(2, 8)  # More frequent activity
    
    def on_start(self):
        """Initialize premium user with enterprise subscription."""
        super().on_start()
        self.user_data["subscription_tier"] = "premium"
    
    @task(15)  # Premium users create more valuations
    def create_premium_valuation_request(self):
        """Create premium valuation with advanced options."""
        if not self.auth_token:
            return
        
        company_data = self._generate_company_data()
        company_data.update({
            "analysis_type": "premium",
            "include_peer_analysis": True,
            "risk_assessment_level": "comprehensive",
            "custom_parameters": {
                "wacc_adjustment": random.uniform(-0.02, 0.02),
                "terminal_growth_override": random.uniform(0.02, 0.05),
                "beta_adjustment": random.uniform(-0.3, 0.3)
            }
        })
        
        with self.client.post(
            "/api/v1/valuations",
            json=company_data,
            headers=self.auth_headers,
            catch_response=True
        ) as response:
            if response.status_code == 201:
                result = response.json()
                job_id = result.get("job_id")
                if job_id:
                    self.valuation_jobs.append({
                        "job_id": job_id,
                        "created_at": time.time(),
                        "status": "queued"
                    })
                response.success()
            else:
                response.failure(f"Premium valuation creation failed: {response.status_code}")
    
    @task(5)
    def bulk_market_data_request(self):
        """Premium feature: bulk market data requests."""
        if not self.auth_token:
            return
        
        tickers = ["CBA", "WBC", "ANZ", "BHP", "CSL", "WES", "WOW", "TLS", "RIO", "MQG"]
        
        bulk_request = {
            "tickers": tickers,
            "include_historical": True,
            "historical_period": "1Y"
        }
        
        with self.client.post(
            "/api/v1/market-data/bulk",
            json=bulk_request,
            headers=self.auth_headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Bulk market data failed: {response.status_code}")
    
    @task(3)
    def advanced_analytics_request(self):
        """Premium feature: advanced analytics."""
        if not self.auth_token:
            return
        
        analytics_request = {
            "analysis_type": "scenario_analysis",
            "scenarios": ["bull_case", "base_case", "bear_case"],
            "sensitivity_variables": ["wacc", "terminal_growth", "revenue_growth"]
        }
        
        with self.client.post(
            "/api/v1/analytics/advanced",
            json=analytics_request,
            headers=self.auth_headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Advanced analytics failed: {response.status_code}")


class APIOnlyUser(FastHttpUser):
    """API-only user for testing pure API performance without browser overhead."""
    
    weight = 2
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize API user."""
        self.api_key = "test-api-key-12345"  # Mock API key
    
    @task(20)
    def api_create_valuation(self):
        """API endpoint for creating valuations."""
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        
        valuation_data = {
            "company_name": f"API Test Corp {random.randint(1000, 9999)}",
            "industry_sector": "Technology",
            "expected_ipo_date": "2024-12-31",
            "shares_on_issue": 50000000,
            "expected_offer_price_range": {"min": 2.0, "max": 4.0}
        }
        
        with self.client.post(
            "/api/v1/valuations",
            json=valuation_data,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(f"API valuation failed: {response.status_code}")
    
    @task(10)
    def api_get_market_data(self):
        """API endpoint for market data."""
        headers = {"X-API-Key": self.api_key}
        ticker = random.choice(["CBA", "WBC", "ANZ", "BHP"])
        
        with self.client.get(
            f"/api/v1/market-data/companies/{ticker}",
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"API market data failed: {response.status_code}")


# Custom test scenarios for specific load patterns
class BlackFridayScenario(IPOValuationUser):
    """Simulates high-traffic scenarios like product launches or market events."""
    
    weight = 1
    wait_time = between(1, 2)  # Very high frequency
    
    @task(25)
    def high_frequency_requests(self):
        """High frequency valuation requests during peak events."""
        super().create_valuation_request()
    
    @task(15)
    def rapid_status_checks(self):
        """Rapid status checking during high-activity periods."""
        super().check_valuation_status()


# Performance monitoring and custom metrics
@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize performance monitoring."""
    if not hasattr(environment, 'stats'):
        return
    
    # Custom metrics tracking
    environment.custom_metrics = {
        'valuation_creation_time': [],
        'report_download_time': [],
        'api_response_times': {}
    }


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    """Track custom metrics for specific endpoints."""
    environment = context.get('environment') if context else None
    if not environment or not hasattr(environment, 'custom_metrics'):
        return
    
    # Track specific endpoint performance
    if name.startswith('/api/v1/valuations') and request_type == 'POST':
        environment.custom_metrics['valuation_creation_time'].append(response_time)
    elif name.startswith('/api/v1/reports/') and request_type == 'GET':
        environment.custom_metrics['report_download_time'].append(response_time)
    
    # Track API response times by endpoint
    endpoint = name.split('?')[0]  # Remove query parameters
    if endpoint not in environment.custom_metrics['api_response_times']:
        environment.custom_metrics['api_response_times'][endpoint] = []
    environment.custom_metrics['api_response_times'][endpoint].append(response_time)


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Generate performance report when test ends."""
    if not hasattr(environment, 'custom_metrics'):
        return
    
    metrics = environment.custom_metrics
    
    # Generate performance report
    report = {
        'test_summary': {
            'total_requests': environment.stats.total.num_requests,
            'total_failures': environment.stats.total.num_failures,
            'average_response_time': environment.stats.total.avg_response_time,
            'requests_per_second': environment.stats.total.current_rps,
        },
        'custom_metrics': {}
    }
    
    # Calculate custom metrics
    if metrics['valuation_creation_time']:
        report['custom_metrics']['valuation_creation'] = {
            'avg_time': sum(metrics['valuation_creation_time']) / len(metrics['valuation_creation_time']),
            'max_time': max(metrics['valuation_creation_time']),
            'min_time': min(metrics['valuation_creation_time'])
        }
    
    if metrics['report_download_time']:
        report['custom_metrics']['report_download'] = {
            'avg_time': sum(metrics['report_download_time']) / len(metrics['report_download_time']),
            'max_time': max(metrics['report_download_time']),
            'min_time': min(metrics['report_download_time'])
        }
    
    # API endpoint performance summary
    report['custom_metrics']['api_endpoints'] = {}
    for endpoint, times in metrics['api_response_times'].items():
        if times:
            report['custom_metrics']['api_endpoints'][endpoint] = {
                'avg_time': sum(times) / len(times),
                'request_count': len(times)
            }
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"load_test_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n=== Load Test Performance Report ===")
    print(f"Total Requests: {report['test_summary']['total_requests']}")
    print(f"Total Failures: {report['test_summary']['total_failures']}")
    print(f"Average Response Time: {report['test_summary']['average_response_time']:.2f}ms")
    print(f"Requests/Second: {report['test_summary']['requests_per_second']:.2f}")
    
    if 'valuation_creation' in report['custom_metrics']:
        vc_metrics = report['custom_metrics']['valuation_creation']
        print(f"Valuation Creation - Avg: {vc_metrics['avg_time']:.2f}ms, Max: {vc_metrics['max_time']:.2f}ms")
    
    print(f"Detailed report saved to: {report_file}")


# Test configuration classes for different scenarios
class LoadTestConfig:
    """Configuration for different load test scenarios."""
    
    SMOKE_TEST = {
        'users': 5,
        'spawn_rate': 1,
        'duration': '2m'
    }
    
    LOAD_TEST = {
        'users': 50,
        'spawn_rate': 5,
        'duration': '10m'
    }
    
    STRESS_TEST = {
        'users': 200,
        'spawn_rate': 10,
        'duration': '15m'
    }
    
    SPIKE_TEST = {
        'users': 500,
        'spawn_rate': 50,
        'duration': '5m'
    }
    
    ENDURANCE_TEST = {
        'users': 100,
        'spawn_rate': 5,
        'duration': '60m'
    }


if __name__ == "__main__":
    """
    Run load tests with different configurations.
    
    Usage examples:
    - Smoke test: locust -f load_testing.py --users 5 --spawn-rate 1 -t 2m
    - Load test: locust -f load_testing.py --users 50 --spawn-rate 5 -t 10m
    - Stress test: locust -f load_testing.py --users 200 --spawn-rate 10 -t 15m
    """
    import sys
    
    print("IPO Valuation Platform Load Testing Suite")
    print("=========================================")
    print("Available test scenarios:")
    print("1. Smoke Test: 5 users, 2 minutes")
    print("2. Load Test: 50 users, 10 minutes") 
    print("3. Stress Test: 200 users, 15 minutes")
    print("4. Spike Test: 500 users, 5 minutes")
    print("5. Endurance Test: 100 users, 60 minutes")
    print("\nRun with: locust -f load_testing.py [options]")
    print("Or use the web UI: locust -f load_testing.py --host http://localhost:8000")