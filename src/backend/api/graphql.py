"""
GraphQL endpoint for complex queries
"""

import strawberry
from strawberry.fastapi import GraphQLRouter
from typing import List, Optional
from datetime import datetime

# GraphQL Types
@strawberry.type
class Company:
    id: int
    name: str
    asx_code: Optional[str]
    industry: Optional[str]
    market_cap: Optional[float]
    created_at: datetime

@strawberry.type
class ValuationResult:
    id: int
    company: Company
    value_per_share: Optional[float]
    target_price: Optional[float]
    recommendation: str
    created_at: datetime

@strawberry.type
class User:
    id: int
    email: str
    first_name: str
    last_name: str
    role: str

# GraphQL Queries
@strawberry.type
class Query:
    @strawberry.field
    def companies(self, search: Optional[str] = None, limit: int = 10) -> List[Company]:
        """Search companies with complex filtering"""
        # Mock data - would implement actual database queries
        return [
            Company(
                id=1,
                name="Commonwealth Bank",
                asx_code="CBA",
                industry="Banking",
                market_cap=150000000000.0,
                created_at=datetime.now()
            )
        ]
    
    @strawberry.field
    def valuation_results(
        self, 
        company_id: Optional[int] = None,
        user_id: Optional[int] = None
    ) -> List[ValuationResult]:
        """Get valuation results with complex filtering"""
        # Mock data
        return []
    
    @strawberry.field
    def company_comparisons(
        self,
        company_ids: List[int],
        metrics: List[str]
    ) -> List[Company]:
        """Compare multiple companies across metrics"""
        return []

# GraphQL Mutations
@strawberry.type
class Mutation:
    @strawberry.field
    def create_company(self, name: str, asx_code: Optional[str] = None) -> Company:
        """Create a new company"""
        return Company(
            id=999,
            name=name,
            asx_code=asx_code,
            industry=None,
            market_cap=None,
            created_at=datetime.now()
        )

# Create GraphQL schema
schema = strawberry.Schema(query=Query, mutation=Mutation)

# Create GraphQL router
graphql_app = GraphQLRouter(schema, path="/graphql")