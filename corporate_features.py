"""
Corporate Tier Features Module
Advanced features for business and government use
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import sqlite3
from datetime import datetime, timedelta
import hashlib
import secrets
from pathlib import Path

class CorporateManager:
    """Manages corporate tier features and functionality"""
    
    def __init__(self):
        """Initialize corporate manager"""
        self.setup_database()
        self.pricing_tiers = {
            'basic': {
                'name': 'Basic',
                'price': 9.99,
                'features': ['Standard Analysis', '100 Analyses/month', 'Email Support'],
                'limits': {'analyses_per_month': 100, 'users': 1, 'api_calls': 1000}
            },
            'professional': {
                'name': 'Professional',
                'price': 29.99,
                'features': ['Advanced Analysis', '1000 Analyses/month', 'Priority Support', 'API Access'],
                'limits': {'analyses_per_month': 1000, 'users': 5, 'api_calls': 10000}
            },
            'enterprise': {
                'name': 'Enterprise',
                'price': 99.99,
                'features': ['Complete Analysis', 'Unlimited Analyses', '24/7 Support', 'Custom Integration', 'White Label'],
                'limits': {'analyses_per_month': float('inf'), 'users': 50, 'api_calls': float('inf')}
            },
            'government': {
                'name': 'Government',
                'price': 'Custom',
                'features': ['Regulatory Compliance', 'Audit Trail', 'Custom Reports', 'On-Premise Option'],
                'limits': {'analyses_per_month': float('inf'), 'users': 200, 'api_calls': float('inf')}
            }
        }
    
    def setup_database(self):
        """Setup database for corporate features"""
        self.db_path = Path("spotscan_corporate.db")
        
        # Create tables if they don't exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                company TEXT NOT NULL,
                tier TEXT NOT NULL,
                api_key TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                analyses_used INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Analysis logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                food_type TEXT,
                analysis_type TEXT,
                results TEXT,
                confidence REAL,
                processing_time REAL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # API usage table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                endpoint TEXT,
                response_time REAL,
                status_code INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Compliance reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                report_type TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                content TEXT,
                hash TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def display_corporate_dashboard(self):
        """Display corporate dashboard"""
        st.title("Spotscan Corporate Dashboard")
        
        # Authentication check
        if not self.check_authentication():
            self.display_login()
            return
        
        # Sidebar navigation
        self.display_sidebar_navigation()
        
        # Main content based on selection
        page = st.session_state.get('corporate_page', 'overview')
        
        if page == 'overview':
            self.display_overview()
        elif page == 'analytics':
            self.display_analytics()
        elif page == 'api_management':
            self.display_api_management()
        elif page == 'compliance':
            self.display_compliance()
        elif page == 'billing':
            self.display_billing()
        elif page == 'team_management':
            self.display_team_management()
        elif page == 'custom_reports':
            self.display_custom_reports()
    
    def check_authentication(self) -> bool:
        """Check if user is authenticated"""
        return st.session_state.get('authenticated', False)
    
    def display_login(self):
        """Display login interface"""
        st.subheader("Corporate Login")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", type="primary"):
                if self.authenticate_user(email, password):
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        with col2:
            st.subheader("New to Spotscan Corporate?")
            st.write("Get started with our powerful food analysis platform.")
            
            if st.button("Sign Up for Free Trial"):
                st.info("Contact sales@spotscan.ai for enterprise trial access")
    
    def authenticate_user(self, email: str, password: str) -> bool:
        """Authenticate user (simplified for demo)"""
        # In production, this would use proper authentication
        return email == "demo@spotscan.ai" and password == "demo123"
    
    def display_sidebar_navigation(self):
        """Display sidebar navigation"""
        with st.sidebar:
            st.title("Corporate Portal")
            
            # User info
            email = st.session_state.get('user_email', 'Unknown')
            st.write(f"Logged in as: {email}")
            
            # Navigation menu
            pages = {
                'overview': 'Overview',
                'analytics': 'Analytics',
                'api_management': 'API Management',
                'compliance': 'Compliance',
                'billing': 'Billing',
                'team_management': 'Team Management',
                'custom_reports': 'Custom Reports'
            }
            
            for page_key, page_name in pages.items():
                if st.button(page_name, key=f"nav_{page_key}"):
                    st.session_state.corporate_page = page_key
                    st.rerun()
            
            st.sidebar.markdown("---")
            
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.user_email = None
                st.rerun()
    
    def display_overview(self):
        """Display overview dashboard"""
        st.header("Corporate Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", self.get_total_analyses())
        
        with col2:
            st.metric("API Calls", self.get_api_calls())
        
        with col3:
            st.metric("Team Members", self.get_team_size())
        
        with col4:
            st.metric("Compliance Score", "98%")
        
        # Usage charts
        st.subheader("Usage Analytics")
        
        # Analysis trends
        analysis_data = self.get_analysis_trends()
        if analysis_data:
            st.line_chart(analysis_data)
        
        # Recent activity
        st.subheader("Recent Activity")
        recent_activity = self.get_recent_activity()
        
        if recent_activity:
            for activity in recent_activity[:5]:
                st.write(f"  :calendar: {activity}")
    
    def display_analytics(self):
        """Display analytics dashboard"""
        st.header("Analytics Dashboard")
        
        # Time period selector
        period = st.selectbox("Time Period", ["Last 7 days", "Last 30 days", "Last 90 days", "Custom"])
        
        # Analysis breakdown
        st.subheader("Analysis Breakdown")
        
        # Food type analysis
        food_analysis = self.get_food_type_analysis()
        if food_analysis:
            st.bar_chart(food_analysis)
        
        # Success rates
        st.subheader("Analysis Success Rates")
        
        success_data = self.get_success_rates()
        if success_data:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall Success", f"{success_data['overall']:.1%}")
            with col2:
                st.metric("Average Confidence", f"{success_data['confidence']:.1%}")
        
        # Detailed metrics
        if st.checkbox("Show Detailed Metrics"):
            self.display_detailed_metrics()
    
    def display_api_management(self):
        """Display API management interface"""
        st.header("API Management")
        
        # API key display
        st.subheader("API Keys")
        
        api_key = self.generate_api_key()
        st.code(api_key, language="text")
        
        st.info("Keep your API key secure. Do not share it publicly.")
        
        # API usage statistics
        st.subheader("API Usage Statistics")
        
        usage_data = self.get_api_usage_stats()
        if usage_data:
            st.line_chart(usage_data)
        
        # API documentation
        st.subheader("API Documentation")
        
        st.markdown("""
        ### API Endpoints
        
        **POST /api/v1/analyze**
        Analyze food image
        
        **Parameters:**
        - `image`: Base64 encoded image
        - `analyses`: List of analyses to perform
        - `food_type`: Optional food type hint
        
        **Response:**
        ```json
        {
            "status": "success",
            "results": {...},
            "confidence": 0.95,
            "processing_time": 1.23
        }
        ```
        
        **GET /api/v1/status**
        Check API status and limits
        
        **GET /api/v1/usage**
        Get current usage statistics
        """)
        
        # Rate limits
        st.subheader("Rate Limits")
        
        tier = self.get_user_tier()
        limits = self.pricing_tiers[tier]['limits']
        
        st.write(f"Current tier: {tier}")
        st.write(f"Analyses per month: {limits['analyses_per_month']}")
        st.write(f"API calls per month: {limits['api_calls']}")
        st.write(f"Team members: {limits['users']}")
    
    def display_compliance(self):
        """Display compliance dashboard"""
        st.header("Compliance & Audit Trail")
        
        # Compliance status
        st.subheader("Compliance Status")
        
        compliance_score = self.calculate_compliance_score()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Score", f"{compliance_score['overall']:.1%}")
        
        with col2:
            st.metric("Data Protection", f"{compliance_score['data_protection']:.1%}")
        
        with col3:
            st.metric("Audit Trail", f"{compliance_score['audit_trail']:.1%}")
        
        # Generate compliance report
        st.subheader("Compliance Reports")
        
        report_type = st.selectbox("Report Type", ["Monthly", "Quarterly", "Annual", "Custom"])
        
        if st.button("Generate Report"):
            report = self.generate_compliance_report(report_type)
            st.success(f"{report_type} report generated successfully!")
            
            # Download link
            st.download_button(
                label="Download Report",
                data=report['content'],
                file_name=f"compliance_report_{report_type.lower()}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
        
        # Audit trail
        st.subheader("Audit Trail")
        
        audit_data = self.get_audit_trail()
        if audit_data:
            # Display as table
            df = pd.DataFrame(audit_data)
            st.dataframe(df)
        
        # Data retention policy
        st.subheader("Data Retention Policy")
        
        retention_info = self.get_retention_policy()
        st.info(retention_info)
    
    def display_billing(self):
        """Display billing information"""
        st.header("Billing & Subscription")
        
        # Current plan
        st.subheader("Current Subscription")
        
        current_tier = self.get_user_tier()
        tier_info = self.pricing_tiers[current_tier]
        
        st.write(f"**Plan:** {tier_info['name']}")
        st.write(f"**Price:** ${tier_info['price']}/month")
        
        st.write("**Features:**")
        for feature in tier_info['features']:
            st.write(f"  :white_check_mark: {feature}")
        
        # Usage summary
        st.subheader("Usage Summary")
        
        usage = self.get_current_usage()
        limits = tier_info['limits']
        
        st.write(f"Analyses used: {usage['analyses']}/{limits['analyses_per_month']}")
        st.write(f"API calls: {usage['api_calls']}/{limits['api_calls']}")
        
        # Progress bars
        if limits['analyses_per_month'] != float('inf'):
            progress = usage['analyses'] / limits['analyses_per_month']
            st.progress(progress, f"Analysis Usage: {progress:.1%}")
        
        if limits['api_calls'] != float('inf'):
            progress = usage['api_calls'] / limits['api_calls']
            st.progress(progress, f"API Usage: {progress:.1%}")
        
        # Upgrade options
        st.subheader("Upgrade Options")
        
        if current_tier != 'government':
            for tier_name, tier_data in self.pricing_tiers.items():
                if tier_name != current_tier and tier_name != 'government':
                    with st.expander(f"Upgrade to {tier_data['name']} - ${tier_data['price']}/month"):
                        st.write("**Features:**")
                        for feature in tier_data['features']:
                            st.write(f"  :white_check_mark: {feature}")
                        
                        if st.button(f"Upgrade to {tier_data['name']}", key=f"upgrade_{tier_name}"):
                            st.info(f"Contact sales@spotscan.ai to upgrade to {tier_data['name']}")
        
        # Billing history
        st.subheader("Billing History")
        
        billing_data = self.get_billing_history()
        if billing_data:
            df = pd.DataFrame(billing_data)
            st.dataframe(df)
    
    def display_team_management(self):
        """Display team management interface"""
        st.header("Team Management")
        
        # Current team members
        st.subheader("Team Members")
        
        team_members = self.get_team_members()
        
        if team_members:
            for member in team_members:
                with st.expander(f"{member['name']} - {member['role']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Email:** {member['email']}")
                        st.write(f"**Status:** {member['status']}")
                    
                    with col2:
                        st.write(f"**Last Login:** {member['last_login']}")
                        st.write(f"**Analyses:** {member['analyses_used']}")
                    
                    with col3:
                        if st.button(f"Remove {member['name']}", key=f"remove_{member['id']}"):
                            st.warning("Team member removal requires admin approval")
        else:
            st.info("No team members found")
        
        # Add team member
        st.subheader("Add Team Member")
        
        with st.form("add_member"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Name")
                email = st.text_input("Email")
            
            with col2:
                role = st.selectbox("Role", ["Analyst", "Manager", "Admin"])
                permissions = st.multiselect("Permissions", ["Read", "Write", "Admin"])
            
            if st.form_submit_button("Add Member"):
                if name and email:
                    self.add_team_member(name, email, role, permissions)
                    st.success(f"Added {name} to the team")
                    st.rerun()
                else:
                    st.error("Please fill in all fields")
        
        # Team permissions
        st.subheader("Team Permissions")
        
        permissions_data = self.get_team_permissions()
        if permissions_data:
            st.dataframe(pd.DataFrame(permissions_data))
    
    def display_custom_reports(self):
        """Display custom reports interface"""
        st.header("Custom Reports")
        
        # Report builder
        st.subheader("Report Builder")
        
        with st.form("report_builder"):
            # Report configuration
            report_name = st.text_input("Report Name")
            
            col1, col2 = st.columns(2)
            
            with col1:
                date_range = st.date_range("Date Range", datetime.now() - timedelta(days=30), datetime.now())
                report_type = st.selectbox("Report Type", ["Summary", "Detailed", "Compliance", "Usage"])
            
            with col2:
                metrics = st.multiselect("Metrics", [
                    "Total Analyses", "Success Rate", "API Usage", "Team Activity",
                    "Food Types", "Confidence Scores", "Processing Times"
                ])
                
                format_type = st.selectbox("Format", ["PDF", "Excel", "CSV", "JSON"])
            
            if st.form_submit_button("Generate Report"):
                if report_name and metrics:
                    report = self.generate_custom_report(report_name, date_range, report_type, metrics, format_type)
                    st.success(f"Report '{report_name}' generated successfully!")
                    
                    # Download link
                    st.download_button(
                        label="Download Report",
                        data=report['content'],
                        file_name=f"{report_name}.{format_type.lower()}",
                        mime=self.get_mime_type(format_type)
                    )
                else:
                    st.error("Please fill in all required fields")
        
        # Saved reports
        st.subheader("Saved Reports")
        
        saved_reports = self.get_saved_reports()
        if saved_reports:
            for report in saved_reports:
                with st.expander(f"{report['name']} - {report['created_at']}"):
                    st.write(f"**Type:** {report['type']}")
                    st.write(f"**Format:** {report['format']}")
                    st.write(f"**Size:** {report['size']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Download {report['name']}", key=f"download_{report['id']}"):
                            st.info("Download functionality would be implemented here")
                    
                    with col2:
                        if st.button(f"Delete {report['name']}", key=f"delete_{report['id']}"):
                            st.warning("Report deletion requires confirmation")
        else:
            st.info("No saved reports found")
    
    # Helper methods
    def generate_api_key(self) -> str:
        """Generate API key for user"""
        return secrets.token_urlsafe(32)
    
    def get_user_tier(self) -> str:
        """Get user's subscription tier"""
        return "professional"  # Default for demo
    
    def get_total_analyses(self) -> int:
        """Get total number of analyses"""
        return 1250  # Mock data
    
    def get_api_calls(self) -> int:
        """Get total API calls"""
        return 8750  # Mock data
    
    def get_team_size(self) -> int:
        """Get team size"""
        return 5  # Mock data
    
    def get_analysis_trends(self) -> Dict:
        """Get analysis trends data"""
        return {
            "Mon": 45, "Tue": 52, "Wed": 48, "Thu": 61, "Fri": 58, "Sat": 35, "Sun": 32
        }
    
    def get_recent_activity(self) -> List[str]:
        """Get recent activity"""
        return [
            "Apple analysis completed - 95% confidence",
            "Team member John Doe logged in",
            "API key regenerated",
            "Monthly compliance report generated",
            "New team member Sarah Smith added"
        ]
    
    def get_food_type_analysis(self) -> Dict:
        """Get food type analysis data"""
        return {
            "Fruits": 120, "Vegetables": 85, "Grains": 45, "Proteins": 95, "Beverages": 35
        }
    
    def get_success_rates(self) -> Dict:
        """Get success rate statistics"""
        return {
            "overall": 0.94,
            "confidence": 0.91
        }
    
    def get_api_usage_stats(self) -> Dict:
        """Get API usage statistics"""
        return {
            "Week 1": 1200, "Week 2": 1450, "Week 3": 1380, "Week 4": 1650
        }
    
    def calculate_compliance_score(self) -> Dict:
        """Calculate compliance score"""
        return {
            "overall": 0.98,
            "data_protection": 0.97,
            "audit_trail": 0.99
        }
    
    def generate_compliance_report(self, report_type: str) -> Dict:
        """Generate compliance report"""
        return {
            "content": f"Mock {report_type} compliance report content",
            "hash": hashlib.sha256(f"report_{report_type}".encode()).hexdigest()
        }
    
    def get_audit_trail(self) -> List[Dict]:
        """Get audit trail data"""
        return [
            {"timestamp": "2024-01-15 10:30:00", "action": "Analysis", "user": "john@company.com"},
            {"timestamp": "2024-01-15 11:45:00", "action": "API Call", "user": "api@company.com"},
            {"timestamp": "2024-01-15 14:20:00", "action": "Report Generation", "user": "admin@company.com"}
        ]
    
    def get_retention_policy(self) -> str:
        """Get data retention policy information"""
        return "Analysis data is retained for 90 days, then anonymized. Audit logs are retained for 7 years."
    
    def get_current_usage(self) -> Dict:
        """Get current usage statistics"""
        return {
            "analyses": 750,
            "api_calls": 5400
        }
    
    def get_billing_history(self) -> List[Dict]:
        """Get billing history"""
        return [
            {"date": "2024-01-01", "amount": 29.99, "status": "Paid", "method": "Credit Card"},
            {"date": "2023-12-01", "amount": 29.99, "status": "Paid", "method": "Credit Card"},
            {"date": "2023-11-01", "amount": 29.99, "status": "Paid", "method": "Credit Card"}
        ]
    
    def get_team_members(self) -> List[Dict]:
        """Get team members"""
        return [
            {"id": 1, "name": "John Doe", "email": "john@company.com", "role": "Admin", "status": "Active", "last_login": "2024-01-15", "analyses_used": 150},
            {"id": 2, "name": "Jane Smith", "email": "jane@company.com", "role": "Analyst", "status": "Active", "last_login": "2024-01-14", "analyses_used": 85}
        ]
    
    def add_team_member(self, name: str, email: str, role: str, permissions: List[str]):
        """Add team member (mock implementation)"""
        pass  # Would implement actual team member addition
    
    def get_team_permissions(self) -> List[Dict]:
        """Get team permissions"""
        return [
            {"user": "John Doe", "permissions": ["Read", "Write", "Admin"]},
            {"user": "Jane Smith", "permissions": ["Read", "Write"]}
        ]
    
    def generate_custom_report(self, name: str, date_range: Tuple, report_type: str, metrics: List[str], format_type: str) -> Dict:
        """Generate custom report"""
        return {
            "content": f"Mock custom report: {name}",
            "format": format_type
        }
    
    def get_mime_type(self, format_type: str) -> str:
        """Get MIME type for format"""
        mime_types = {
            "PDF": "application/pdf",
            "Excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "CSV": "text/csv",
            "JSON": "application/json"
        }
        return mime_types.get(format_type, "text/plain")
    
    def get_saved_reports(self) -> List[Dict]:
        """Get saved reports"""
        return [
            {"id": 1, "name": "Monthly Analysis Report", "created_at": "2024-01-01", "type": "Summary", "format": "PDF", "size": "2.3MB"},
            {"id": 2, "name": "Compliance Report Q4", "created_at": "2023-12-31", "type": "Compliance", "format": "PDF", "size": "1.8MB"}
        ]
    
    def display_detailed_metrics(self):
        """Display detailed metrics"""
        st.write("#### Processing Times")
        processing_times = [1.2, 1.5, 1.1, 1.8, 1.3, 1.4, 1.6, 1.2]
        st.line_chart(processing_times)
        
        st.write("#### Confidence Distribution")
        confidence_data = {
            "0.9-1.0": 45, "0.8-0.9": 30, "0.7-0.8": 20, "0.6-0.7": 5
        }
        st.bar_chart(confidence_data)

# Main corporate app function
def run_corporate_app():
    """Run the corporate application"""
    corporate = CorporateManager()
    corporate.display_corporate_dashboard()

if __name__ == "__main__":
    run_corporate_app()
