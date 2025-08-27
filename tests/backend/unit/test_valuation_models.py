"""
Unit tests for valuation models and business logic.

Tests the core valuation models including DCF, comparable company analysis,
and risk assessment models.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

from tests.utils.factories import CompanyFactory, ValuationFactory, MarketDataFactory
from tests.utils.test_data import get_standard_test_scenario


class TestDCFModel:
    """Test Discounted Cash Flow model calculations."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.company_data = CompanyFactory.build(
            annual_revenue=Decimal('50000000'),
            industry_sector='Technology'
        )
        
        self.financial_projections = {
            'revenue_growth_rates': [0.25, 0.20, 0.15, 0.12, 0.10],
            'ebitda_margins': [0.20, 0.22, 0.24, 0.25, 0.25],
            'tax_rate': 0.25,
            'capex_as_pct_revenue': 0.05,
            'working_capital_change': 0.02
        }
        
        self.valuation_assumptions = {
            'wacc': 0.12,
            'terminal_growth_rate': 0.03,
            'forecast_years': 5
        }
    
    def test_dcf_revenue_projections(self):
        """Test revenue projection calculations."""
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        
        dcf_model = DCFModel()
        
        base_revenue = float(self.company_data['annual_revenue'])
        growth_rates = self.financial_projections['revenue_growth_rates']
        
        projections = dcf_model.project_revenue(base_revenue, growth_rates)
        
        # Verify projections structure
        assert len(projections) == len(growth_rates)
        assert all(isinstance(p, (int, float)) for p in projections)
        
        # Verify growth calculations
        expected_year_1 = base_revenue * (1 + growth_rates[0])
        assert abs(projections[0] - expected_year_1) < 0.01
        
        # Verify compound growth
        for i in range(1, len(projections)):
            expected = projections[i-1] * (1 + growth_rates[i])
            assert abs(projections[i] - expected) < 0.01
    
    def test_dcf_ebitda_calculations(self):
        """Test EBITDA projection calculations."""
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        
        dcf_model = DCFModel()
        
        revenues = [60000000, 72000000, 82800000, 92736000, 102009600]
        ebitda_margins = self.financial_projections['ebitda_margins']
        
        ebitda_projections = dcf_model.calculate_ebitda(revenues, ebitda_margins)
        
        # Verify calculations
        assert len(ebitda_projections) == len(revenues)
        
        for i, (revenue, margin) in enumerate(zip(revenues, ebitda_margins)):
            expected_ebitda = revenue * margin
            assert abs(ebitda_projections[i] - expected_ebitda) < 0.01
    
    def test_dcf_free_cash_flow_calculations(self):
        """Test free cash flow calculations."""
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        
        dcf_model = DCFModel()
        
        ebitda_projections = [12000000, 15840000, 19872000, 23184000, 25502400]
        revenues = [60000000, 72000000, 82800000, 92736000, 102009600]
        
        fcf_projections = dcf_model.calculate_free_cash_flow(
            ebitda_projections=ebitda_projections,
            revenues=revenues,
            tax_rate=self.financial_projections['tax_rate'],
            capex_as_pct_revenue=self.financial_projections['capex_as_pct_revenue'],
            working_capital_change=self.financial_projections['working_capital_change']
        )
        
        # Verify structure
        assert len(fcf_projections) == len(ebitda_projections)
        
        # Verify that FCF is less than EBITDA (due to taxes, capex, etc.)
        for fcf, ebitda in zip(fcf_projections, ebitda_projections):
            assert fcf < ebitda
            assert fcf > 0  # Should be positive for healthy company
    
    def test_dcf_terminal_value_calculation(self):
        """Test terminal value calculations."""
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        
        dcf_model = DCFModel()
        
        final_year_fcf = 18500000
        terminal_growth_rate = 0.03
        wacc = 0.12
        
        terminal_value = dcf_model.calculate_terminal_value(
            final_year_fcf, terminal_growth_rate, wacc
        )
        
        # Verify Gordon Growth Model calculation
        expected_terminal_fcf = final_year_fcf * (1 + terminal_growth_rate)
        expected_terminal_value = expected_terminal_fcf / (wacc - terminal_growth_rate)
        
        assert abs(terminal_value - expected_terminal_value) < 1000
    
    def test_dcf_present_value_calculations(self):
        """Test present value discount calculations."""
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        
        dcf_model = DCFModel()
        
        cash_flows = [10000000, 12000000, 14000000, 16000000, 18000000]
        wacc = 0.12
        
        present_values = dcf_model.discount_to_present_value(cash_flows, wacc)
        
        # Verify structure
        assert len(present_values) == len(cash_flows)
        
        # Verify discount calculations
        for i, (cf, pv) in enumerate(zip(cash_flows, present_values)):
            expected_pv = cf / ((1 + wacc) ** (i + 1))
            assert abs(pv - expected_pv) < 0.01
            assert pv < cf  # Present value should be less than future value
    
    def test_dcf_enterprise_value_calculation(self):
        """Test enterprise value calculation."""
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        
        dcf_model = DCFModel()
        
        fcf_present_values = [8928571, 9567901, 9967641, 10179898, 10289203]
        terminal_present_value = 125000000
        
        enterprise_value = dcf_model.calculate_enterprise_value(
            fcf_present_values, terminal_present_value
        )
        
        expected_ev = sum(fcf_present_values) + terminal_present_value
        assert abs(enterprise_value - expected_ev) < 1000
    
    def test_dcf_equity_value_calculation(self):
        """Test equity value and per-share price calculation."""
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        
        dcf_model = DCFModel()
        
        enterprise_value = 173933214
        net_debt = 15000000  # Positive = net debt, negative = net cash
        shares_outstanding = 50000000
        
        equity_value, price_per_share = dcf_model.calculate_equity_value(
            enterprise_value, net_debt, shares_outstanding
        )
        
        expected_equity_value = enterprise_value - net_debt
        expected_price_per_share = expected_equity_value / shares_outstanding
        
        assert abs(equity_value - expected_equity_value) < 1000
        assert abs(price_per_share - expected_price_per_share) < 0.01
    
    def test_dcf_full_valuation_integration(self):
        """Test complete DCF valuation process."""
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        
        dcf_model = DCFModel()
        
        valuation_inputs = {
            'company_data': self.company_data,
            'financial_projections': self.financial_projections,
            'valuation_assumptions': self.valuation_assumptions,
            'net_debt': 10000000,
            'shares_outstanding': 40000000
        }
        
        result = dcf_model.calculate_valuation(valuation_inputs)
        
        # Verify result structure
        assert 'enterprise_value' in result
        assert 'equity_value' in result
        assert 'price_per_share' in result
        assert 'cash_flow_projections' in result
        assert 'assumptions' in result
        
        # Verify reasonable values
        assert result['enterprise_value'] > 0
        assert result['equity_value'] > 0
        assert result['price_per_share'] > 0
        assert result['price_per_share'] < 1000  # Reasonable upper bound
    
    def test_dcf_sensitivity_analysis(self):
        """Test DCF sensitivity to key assumptions."""
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        
        dcf_model = DCFModel()
        
        base_inputs = {
            'company_data': self.company_data,
            'financial_projections': self.financial_projections,
            'valuation_assumptions': self.valuation_assumptions,
            'net_debt': 10000000,
            'shares_outstanding': 40000000
        }
        
        # Test WACC sensitivity
        wacc_scenarios = [0.10, 0.12, 0.14]
        wacc_results = []
        
        for wacc in wacc_scenarios:
            inputs = base_inputs.copy()
            inputs['valuation_assumptions']['wacc'] = wacc
            result = dcf_model.calculate_valuation(inputs)
            wacc_results.append(result['price_per_share'])
        
        # Higher WACC should result in lower valuation
        assert wacc_results[0] > wacc_results[1] > wacc_results[2]
        
        # Test terminal growth sensitivity
        tgr_scenarios = [0.02, 0.03, 0.04]
        tgr_results = []
        
        for tgr in tgr_scenarios:
            inputs = base_inputs.copy()
            inputs['valuation_assumptions']['terminal_growth_rate'] = tgr
            result = dcf_model.calculate_valuation(inputs)
            tgr_results.append(result['price_per_share'])
        
        # Higher terminal growth should result in higher valuation
        assert tgr_results[0] < tgr_results[1] < tgr_results[2]
    
    def test_dcf_error_handling(self):
        """Test DCF model error handling for invalid inputs."""
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        
        dcf_model = DCFModel()
        
        # Test negative revenue
        with pytest.raises(ValueError, match="Revenue must be positive"):
            dcf_model.project_revenue(-1000000, [0.1, 0.1])
        
        # Test WACC >= terminal growth rate
        with pytest.raises(ValueError, match="WACC must be greater than terminal growth rate"):
            dcf_model.calculate_terminal_value(
                final_year_fcf=10000000,
                terminal_growth_rate=0.15,
                wacc=0.12
            )
        
        # Test zero shares outstanding
        with pytest.raises(ValueError, match="Shares outstanding must be positive"):
            dcf_model.calculate_equity_value(
                enterprise_value=100000000,
                net_debt=0,
                shares_outstanding=0
            )


class TestComparableCompanyAnalysis:
    """Test Comparable Company Analysis (CCA) model."""
    
    def setup_method(self):
        """Set up test data for CCA tests."""
        self.target_company = CompanyFactory.build(
            industry_sector='Technology',
            annual_revenue=Decimal('45000000')
        )
        
        # Create peer companies
        self.peer_companies = [
            MarketDataFactory.build(
                industry_sector='Technology',
                market_cap=800000000,
                pe_ratio=18.5,
                ev_ebitda=12.3
            )
            for _ in range(10)
        ]
    
    def test_cca_peer_filtering(self):
        """Test peer company filtering logic."""
        from src.backend.ml_services.valuation_engine.cca_model import CCAModel
        
        cca_model = CCAModel()
        
        # Create diverse market data
        all_companies = [
            MarketDataFactory.build(
                industry_sector='Technology',
                market_cap=500000000
            ),
            MarketDataFactory.build(
                industry_sector='Healthcare',  # Different industry
                market_cap=500000000
            ),
            MarketDataFactory.build(
                industry_sector='Technology',
                market_cap=50000000000  # Too large
            ),
            MarketDataFactory.build(
                industry_sector='Technology',
                market_cap=10000000  # Too small
            )
        ]
        
        filters = {
            'industry_sector': 'Technology',
            'min_market_cap': 100000000,
            'max_market_cap': 10000000000
        }
        
        filtered_peers = cca_model.filter_peer_companies(all_companies, filters)
        
        # Should only include appropriate Technology companies
        assert len(filtered_peers) == 1
        assert filtered_peers[0]['industry_sector'] == 'Technology'
        assert 100000000 <= filtered_peers[0]['market_cap'] <= 10000000000
    
    def test_cca_multiple_calculations(self):
        """Test calculation of valuation multiples."""
        from src.backend.ml_services.valuation_engine.cca_model import CCAModel
        
        cca_model = CCAModel()
        
        peer_data = [
            {
                'market_cap': 1000000000,
                'revenue': 200000000,
                'ebitda': 40000000,
                'net_income': 25000000,
                'pe_ratio': 15.0,
                'ev_ebitda': 12.5
            }
        ]
        
        multiples = cca_model.calculate_multiples(peer_data)
        
        # Verify multiple calculations
        assert 'price_to_sales' in multiples
        assert 'price_to_earnings' in multiples
        assert 'ev_to_ebitda' in multiples
        assert 'ev_to_sales' in multiples
        
        # Verify calculations
        expected_ps = peer_data[0]['market_cap'] / peer_data[0]['revenue']
        assert abs(multiples['price_to_sales'][0] - expected_ps) < 0.01
    
    def test_cca_statistical_measures(self):
        """Test statistical measures calculation for multiples."""
        from src.backend.ml_services.valuation_engine.cca_model import CCAModel
        
        cca_model = CCAModel()
        
        # Create test multiples data
        multiples_data = {
            'pe_ratio': [12.0, 15.0, 18.0, 20.0, 25.0],
            'ev_ebitda': [8.0, 10.0, 12.0, 14.0, 16.0]
        }
        
        statistics = cca_model.calculate_statistics(multiples_data)
        
        # Verify statistical measures
        assert 'pe_ratio' in statistics
        assert 'ev_ebitda' in statistics
        
        pe_stats = statistics['pe_ratio']
        assert 'mean' in pe_stats
        assert 'median' in pe_stats
        assert 'percentile_25' in pe_stats
        assert 'percentile_75' in pe_stats
        assert 'std_dev' in pe_stats
        
        # Verify calculations
        assert pe_stats['median'] == 18.0  # Middle value
        assert pe_stats['mean'] == 18.0   # Average of [12,15,18,20,25]
    
    def test_cca_valuation_application(self):
        """Test applying peer multiples to target company."""
        from src.backend.ml_services.valuation_engine.cca_model import CCAModel
        
        cca_model = CCAModel()
        
        target_metrics = {
            'revenue': 45000000,
            'ebitda': 9000000,
            'net_income': 6000000
        }
        
        peer_multiples = {
            'price_to_sales': {'median': 4.5, 'mean': 4.8},
            'ev_to_ebitda': {'median': 12.0, 'mean': 12.5},
            'price_to_earnings': {'median': 18.0, 'mean': 19.0}
        }
        
        valuations = cca_model.apply_multiples(target_metrics, peer_multiples)
        
        # Verify valuation calculations
        assert 'price_to_sales' in valuations
        assert 'ev_to_ebitda' in valuations
        assert 'price_to_earnings' in valuations
        
        # Verify calculations
        expected_ps_valuation = target_metrics['revenue'] * peer_multiples['price_to_sales']['median']
        assert abs(valuations['price_to_sales']['median'] - expected_ps_valuation) < 1000
    
    def test_cca_weighted_valuation(self):
        """Test weighted average valuation calculation."""
        from src.backend.ml_services.valuation_engine.cca_model import CCAModel
        
        cca_model = CCAModel()
        
        method_valuations = {
            'price_to_sales': {'median': 200000000},
            'ev_to_ebitda': {'median': 180000000},
            'price_to_earnings': {'median': 220000000}
        }
        
        weights = {
            'price_to_sales': 0.3,
            'ev_to_ebitda': 0.4,
            'price_to_earnings': 0.3
        }
        
        weighted_valuation = cca_model.calculate_weighted_valuation(
            method_valuations, weights
        )
        
        # Verify weighted calculation
        expected_weighted = (
            200000000 * 0.3 +
            180000000 * 0.4 +
            220000000 * 0.3
        )
        
        assert abs(weighted_valuation - expected_weighted) < 1000
    
    def test_cca_full_analysis(self):
        """Test complete CCA analysis process."""
        from src.backend.ml_services.valuation_engine.cca_model import CCAModel
        
        cca_model = CCAModel()
        
        analysis_inputs = {
            'target_company': self.target_company,
            'peer_companies': self.peer_companies,
            'target_financials': {
                'revenue': 45000000,
                'ebitda': 9000000,
                'net_income': 5500000
            }
        }
        
        result = cca_model.perform_analysis(analysis_inputs)
        
        # Verify result structure
        assert 'peer_statistics' in result
        assert 'valuation_methods' in result
        assert 'weighted_valuation' in result
        assert 'confidence_metrics' in result
        
        # Verify reasonable values
        assert result['weighted_valuation'] > 0
        assert 'peer_count' in result['confidence_metrics']
        assert result['confidence_metrics']['peer_count'] > 0


class TestRiskAssessmentModel:
    """Test risk assessment model for valuations."""
    
    def setup_method(self):
        """Set up test data for risk assessment tests."""
        self.company_profile = {
            'industry_sector': 'Technology',
            'revenue': 45000000,
            'employee_count': 150,
            'founded_year': 2018,
            'debt_to_equity': 0.3,
            'current_ratio': 2.1
        }
        
        self.market_conditions = {
            'market_volatility': 0.18,
            'sector_performance': 0.12,
            'economic_indicators': {
                'gdp_growth': 0.025,
                'inflation_rate': 0.035,
                'interest_rates': 0.045
            }
        }
    
    def test_risk_score_calculation(self):
        """Test overall risk score calculation."""
        from src.backend.ml_services.valuation_engine.risk_model import RiskAssessmentModel
        
        risk_model = RiskAssessmentModel()
        
        risk_factors = {
            'market_risk': 0.6,
            'financial_risk': 0.4,
            'operational_risk': 0.3,
            'regulatory_risk': 0.2
        }
        
        overall_score = risk_model.calculate_overall_risk_score(risk_factors)
        
        # Verify score is reasonable
        assert 0 <= overall_score <= 10
        assert isinstance(overall_score, (int, float))
    
    def test_market_risk_assessment(self):
        """Test market risk assessment."""
        from src.backend.ml_services.valuation_engine.risk_model import RiskAssessmentModel
        
        risk_model = RiskAssessmentModel()
        
        market_risk = risk_model.assess_market_risk(
            self.company_profile, self.market_conditions
        )
        
        # Verify risk assessment structure
        assert 'score' in market_risk
        assert 'factors' in market_risk
        assert 0 <= market_risk['score'] <= 1
        
        # Should identify key market risk factors
        assert len(market_risk['factors']) > 0
    
    def test_financial_risk_assessment(self):
        """Test financial risk assessment."""
        from src.backend.ml_services.valuation_engine.risk_model import RiskAssessmentModel
        
        risk_model = RiskAssessmentModel()
        
        financial_metrics = {
            'debt_to_equity': 0.3,
            'current_ratio': 2.1,
            'interest_coverage': 8.5,
            'revenue_growth_volatility': 0.15
        }
        
        financial_risk = risk_model.assess_financial_risk(financial_metrics)
        
        # Verify assessment
        assert 'score' in financial_risk
        assert 'factors' in financial_risk
        assert 0 <= financial_risk['score'] <= 1
    
    def test_operational_risk_assessment(self):
        """Test operational risk assessment."""
        from src.backend.ml_services.valuation_engine.risk_model import RiskAssessmentModel
        
        risk_model = RiskAssessmentModel()
        
        operational_factors = {
            'key_person_dependency': True,
            'customer_concentration': 0.4,  # Top 5 customers = 40% of revenue
            'supplier_concentration': 0.3,
            'technology_obsolescence_risk': 'medium',
            'regulatory_compliance_issues': False
        }
        
        operational_risk = risk_model.assess_operational_risk(operational_factors)
        
        # Verify assessment
        assert 'score' in operational_risk
        assert 'factors' in operational_risk
        assert 0 <= operational_risk['score'] <= 1
    
    def test_industry_specific_risk_factors(self):
        """Test industry-specific risk factor identification."""
        from src.backend.ml_services.valuation_engine.risk_model import RiskAssessmentModel
        
        risk_model = RiskAssessmentModel()
        
        # Test technology industry risks
        tech_risks = risk_model.get_industry_risk_factors('Technology')
        
        expected_tech_risks = [
            'technology_obsolescence',
            'cyber_security_threats',
            'talent_competition',
            'regulatory_changes'
        ]
        
        for risk in expected_tech_risks:
            assert any(risk in factor for factor in tech_risks)
        
        # Test healthcare industry risks
        healthcare_risks = risk_model.get_industry_risk_factors('Healthcare')
        
        expected_healthcare_risks = [
            'regulatory_approval',
            'clinical_trial_risks',
            'reimbursement_changes'
        ]
        
        for risk in expected_healthcare_risks:
            assert any(risk in factor for factor in healthcare_risks)
    
    def test_risk_mitigation_recommendations(self):
        """Test risk mitigation recommendation generation."""
        from src.backend.ml_services.valuation_engine.risk_model import RiskAssessmentModel
        
        risk_model = RiskAssessmentModel()
        
        identified_risks = [
            {
                'category': 'financial',
                'factor': 'high_debt_levels',
                'severity': 'high',
                'score': 0.8
            },
            {
                'category': 'operational',
                'factor': 'key_person_dependency',
                'severity': 'medium',
                'score': 0.6
            }
        ]
        
        recommendations = risk_model.generate_mitigation_recommendations(identified_risks)
        
        # Verify recommendations
        assert len(recommendations) > 0
        for rec in recommendations:
            assert 'risk_factor' in rec
            assert 'recommendation' in rec
            assert 'priority' in rec
    
    def test_risk_impact_on_valuation(self):
        """Test how risk assessment impacts valuation multiples."""
        from src.backend.ml_services.valuation_engine.risk_model import RiskAssessmentModel
        
        risk_model = RiskAssessmentModel()
        
        base_valuation = 150000000
        risk_score = 7.5  # High risk
        
        adjusted_valuation = risk_model.apply_risk_adjustment(base_valuation, risk_score)
        
        # High risk should reduce valuation
        assert adjusted_valuation < base_valuation
        
        # Test low risk scenario
        low_risk_score = 2.0
        low_risk_valuation = risk_model.apply_risk_adjustment(base_valuation, low_risk_score)
        
        # Low risk should have minimal impact or slight premium
        assert low_risk_valuation >= adjusted_valuation
    
    def test_scenario_analysis_integration(self):
        """Test integration of risk assessment with scenario analysis."""
        from src.backend.ml_services.valuation_engine.risk_model import RiskAssessmentModel
        
        risk_model = RiskAssessmentModel()
        
        scenarios = {
            'bull_case': {'probability': 0.25, 'risk_multiplier': 0.8},
            'base_case': {'probability': 0.50, 'risk_multiplier': 1.0},
            'bear_case': {'probability': 0.25, 'risk_multiplier': 1.3}
        }
        
        base_risk_score = 5.0
        scenario_risks = risk_model.calculate_scenario_risks(base_risk_score, scenarios)
        
        # Verify scenario risk calculations
        assert len(scenario_risks) == 3
        assert scenario_risks['bull_case'] < base_risk_score
        assert scenario_risks['bear_case'] > base_risk_score
        assert scenario_risks['base_case'] == base_risk_score


@pytest.mark.integration
class TestValuationModelIntegration:
    """Integration tests for valuation models working together."""
    
    def test_combined_valuation_approach(self):
        """Test combining DCF and CCA approaches."""
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        from src.backend.ml_services.valuation_engine.cca_model import CCAModel
        from src.backend.ml_services.valuation_engine.risk_model import RiskAssessmentModel
        
        # Set up test scenario
        scenario = get_standard_test_scenario()
        company = scenario['company']
        peer_data = scenario['peer_data']
        
        # Run DCF analysis
        dcf_model = DCFModel()
        dcf_inputs = {
            'company_data': company,
            'financial_projections': {
                'revenue_growth_rates': [0.20, 0.15, 0.12, 0.10, 0.08],
                'ebitda_margins': [0.18, 0.20, 0.22, 0.23, 0.24],
                'tax_rate': 0.25,
                'capex_as_pct_revenue': 0.04,
                'working_capital_change': 0.02
            },
            'valuation_assumptions': {
                'wacc': 0.11,
                'terminal_growth_rate': 0.03,
                'forecast_years': 5
            },
            'net_debt': 5000000,
            'shares_outstanding': int(company['shares_to_be_issued'])
        }
        
        dcf_result = dcf_model.calculate_valuation(dcf_inputs)
        
        # Run CCA analysis
        cca_model = CCAModel()
        cca_inputs = {
            'target_company': company,
            'peer_companies': peer_data,
            'target_financials': {
                'revenue': int(company['annual_revenue']),
                'ebitda': int(company['annual_revenue']) * 0.20,
                'net_income': int(company['annual_revenue']) * 0.12
            }
        }
        
        cca_result = cca_model.perform_analysis(cca_inputs)
        
        # Run risk assessment
        risk_model = RiskAssessmentModel()
        risk_inputs = {
            'industry_sector': company['industry_sector'],
            'revenue': int(company['annual_revenue']),
            'employee_count': company['employee_count'],
            'founded_year': company['founded_year']
        }
        
        risk_assessment = risk_model.assess_overall_risk(risk_inputs)
        
        # Combine results
        combined_valuation = {
            'dcf_valuation': dcf_result['price_per_share'],
            'cca_valuation': cca_result['weighted_valuation'] / int(company['shares_to_be_issued']),
            'risk_score': risk_assessment['overall_score'],
            'confidence_score': min(
                dcf_result.get('confidence', 0.8),
                cca_result.get('confidence_metrics', {}).get('overall_confidence', 0.8)
            )
        }
        
        # Calculate weighted final valuation
        dcf_weight = 0.6
        cca_weight = 0.4
        
        final_price = (
            combined_valuation['dcf_valuation'] * dcf_weight +
            combined_valuation['cca_valuation'] * cca_weight
        )
        
        # Apply risk adjustment
        risk_adjusted_price = risk_model.apply_risk_adjustment(
            final_price, risk_assessment['overall_score']
        )
        
        # Verify final result
        assert risk_adjusted_price > 0
        assert isinstance(risk_adjusted_price, (int, float))
        assert 0.5 <= risk_adjusted_price <= 20.0  # Reasonable IPO price range
        
        # Verify that risk adjustment impacted the valuation
        if risk_assessment['overall_score'] > 5:
            assert risk_adjusted_price < final_price
        
        # Store complete analysis
        complete_analysis = {
            'target_price': risk_adjusted_price,
            'price_range': {
                'low': risk_adjusted_price * 0.85,
                'high': risk_adjusted_price * 1.15
            },
            'dcf_analysis': dcf_result,
            'cca_analysis': cca_result,
            'risk_analysis': risk_assessment,
            'methodology': {
                'dcf_weight': dcf_weight,
                'cca_weight': cca_weight,
                'risk_adjustment_applied': True
            },
            'confidence_score': combined_valuation['confidence_score']
        }
        
        # Verify complete analysis structure
        required_keys = [
            'target_price', 'price_range', 'dcf_analysis',
            'cca_analysis', 'risk_analysis', 'methodology', 'confidence_score'
        ]
        
        for key in required_keys:
            assert key in complete_analysis
        
        return complete_analysis


@pytest.mark.performance
class TestValuationModelPerformance:
    """Performance tests for valuation models."""
    
    def test_dcf_calculation_performance(self):
        """Test DCF calculation performance."""
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        import time
        
        dcf_model = DCFModel()
        
        # Prepare test inputs
        test_inputs = {
            'company_data': CompanyFactory.build(),
            'financial_projections': {
                'revenue_growth_rates': [0.20, 0.15, 0.12, 0.10, 0.08],
                'ebitda_margins': [0.18, 0.20, 0.22, 0.23, 0.24],
                'tax_rate': 0.25,
                'capex_as_pct_revenue': 0.04,
                'working_capital_change': 0.02
            },
            'valuation_assumptions': {
                'wacc': 0.11,
                'terminal_growth_rate': 0.03,
                'forecast_years': 5
            },
            'net_debt': 5000000,
            'shares_outstanding': 50000000
        }
        
        # Measure performance
        start_time = time.time()
        result = dcf_model.calculate_valuation(test_inputs)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        
        # Should complete within reasonable time
        assert calculation_time < 1.0  # Less than 1 second
        assert result is not None
    
    def test_cca_analysis_performance(self):
        """Test CCA analysis performance with large peer group."""
        from src.backend.ml_services.valuation_engine.cca_model import CCAModel
        import time
        
        cca_model = CCAModel()
        
        # Create large peer group
        peer_companies = [MarketDataFactory.build() for _ in range(100)]
        
        test_inputs = {
            'target_company': CompanyFactory.build(),
            'peer_companies': peer_companies,
            'target_financials': {
                'revenue': 45000000,
                'ebitda': 9000000,
                'net_income': 5500000
            }
        }
        
        # Measure performance
        start_time = time.time()
        result = cca_model.perform_analysis(test_inputs)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        
        # Should complete within reasonable time even with large peer group
        assert calculation_time < 2.0  # Less than 2 seconds
        assert result is not None