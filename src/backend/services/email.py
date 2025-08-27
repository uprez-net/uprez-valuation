"""
Email service for sending notifications
"""

from typing import Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from ..core.config import settings


class EmailService:
    """Service for sending emails"""
    
    @staticmethod
    async def send_welcome_email(email: str, first_name: str):
        """Send welcome email to new users"""
        subject = f"Welcome to {settings.PROJECT_NAME}"
        
        html_content = f"""
        <html>
        <body>
            <h2>Welcome to {settings.PROJECT_NAME}, {first_name}!</h2>
            <p>Thank you for joining our IPO valuation platform.</p>
            <p>You can now start analyzing companies and creating valuations.</p>
            <p>Best regards,<br>The {settings.PROJECT_NAME} Team</p>
        </body>
        </html>
        """
        
        await EmailService._send_email(email, subject, html_content)
    
    @staticmethod
    async def send_password_reset_email(email: str, first_name: str, reset_token: str):
        """Send password reset email"""
        subject = "Password Reset Request"
        
        reset_url = f"{settings.SERVER_HOST}/reset-password?token={reset_token}"
        
        html_content = f"""
        <html>
        <body>
            <h2>Password Reset Request</h2>
            <p>Hello {first_name},</p>
            <p>You requested a password reset for your {settings.PROJECT_NAME} account.</p>
            <p><a href="{reset_url}">Click here to reset your password</a></p>
            <p>This link will expire in 1 hour.</p>
            <p>If you didn't request this, please ignore this email.</p>
        </body>
        </html>
        """
        
        await EmailService._send_email(email, subject, html_content)
    
    @staticmethod
    async def send_verification_email(email: str, first_name: str, verification_token: str):
        """Send email verification"""
        subject = "Verify Your Email Address"
        
        verify_url = f"{settings.SERVER_HOST}/verify-email?token={verification_token}"
        
        html_content = f"""
        <html>
        <body>
            <h2>Email Verification</h2>
            <p>Hello {first_name},</p>
            <p>Please verify your email address by clicking the link below:</p>
            <p><a href="{verify_url}">Verify Email Address</a></p>
            <p>This link will expire in 24 hours.</p>
        </body>
        </html>
        """
        
        await EmailService._send_email(email, subject, html_content)
    
    @staticmethod
    async def _send_email(to_email: str, subject: str, html_content: str):
        """Send email using SMTP"""
        try:
            if not all([settings.SMTP_HOST, settings.SMTP_USER, settings.SMTP_PASSWORD]):
                print(f"EMAIL: {subject} to {to_email}")  # Mock for development
                return
            
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = settings.EMAILS_FROM_EMAIL
            msg["To"] = to_email
            
            # Add HTML content
            html_part = MIMEText(html_content, "html")
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                if settings.SMTP_TLS:
                    server.starttls()
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.send_message(msg)
                
        except Exception as e:
            print(f"Failed to send email: {e}")  # Log error