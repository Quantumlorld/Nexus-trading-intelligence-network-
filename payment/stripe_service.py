"""
Nexus Trading System - Stripe Payment Service
Enterprise payment processing for subscriptions and trading fees
"""

import stripe
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class PlanType(Enum):
    """Subscription plan types"""
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    TRADER = "trader"

class PaymentStatus(Enum):
    """Payment status types"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"

@dataclass
class SubscriptionPlan:
    """Subscription plan configuration"""
    plan_id: str
    name: str
    price: float
    currency: str
    interval: str  # month, year
    features: List[str]
    max_trades_per_day: int
    max_strategies: int
    api_access: bool
    priority_support: bool
    stripe_price_id: Optional[str] = None

class StripeService:
    """Stripe payment service for enterprise features"""
    
    def __init__(self, api_key: str, webhook_secret: str = None):
        self.api_key = api_key
        self.webhook_secret = webhook_secret
        stripe.api_key = api_key
        
        # Initialize subscription plans
        self.plans = self._initialize_plans()
        
        logger.info("Stripe service initialized")
    
    def _initialize_plans(self) -> Dict[str, SubscriptionPlan]:
        """Initialize subscription plans"""
        return {
            PlanType.BASIC.value: SubscriptionPlan(
                plan_id="basic_monthly",
                name="Basic Trader",
                price=29.99,
                currency="usd",
                interval="month",
                features=[
                    "Up to 10 trades per day",
                    "Basic analytics",
                    "Email support",
                    "Mobile app access"
                ],
                max_trades_per_day=10,
                max_strategies=3,
                api_access=False,
                priority_support=False
            ),
            PlanType.PROFESSIONAL.value: SubscriptionPlan(
                plan_id="professional_monthly",
                name="Professional Trader",
                price=99.99,
                currency="usd",
                interval="month",
                features=[
                    "Unlimited trades",
                    "Advanced analytics",
                    "AI-powered signals",
                    "Priority support",
                    "API access",
                    "Custom strategies"
                ],
                max_trades_per_day=float('inf'),
                max_strategies=10,
                api_access=True,
                priority_support=True
            ),
            PlanType.ENTERPRISE.value: SubscriptionPlan(
                plan_id="enterprise_monthly",
                name="Enterprise Trading",
                price=299.99,
                currency="usd",
                interval="month",
                features=[
                    "Unlimited everything",
                    "White-label solutions",
                    "Dedicated support",
                    "Custom integrations",
                    "Advanced AI features",
                    "Risk management tools",
                    "Compliance reporting"
                ],
                max_trades_per_day=float('inf'),
                max_strategies=float('inf'),
                api_access=True,
                priority_support=True
            ),
            PlanType.TRADER.value: SubscriptionPlan(
                plan_id="trader_monthly",
                name="Active Trader",
                price=49.99,
                currency="usd",
                interval="month",
                features=[
                    "Up to 50 trades per day",
                    "Real-time signals",
                    "Performance analytics",
                    "Email support",
                    "Mobile app access"
                ],
                max_trades_per_day=50,
                max_strategies=5,
                api_access=False,
                priority_support=False
            )
        }
    
    async def create_customer(self, user_id: str, email: str, name: str = None) -> Dict[str, Any]:
        """Create a Stripe customer"""
        try:
            customer_data = {
                "email": email,
                "metadata": {
                    "user_id": user_id,
                    "created_at": datetime.utcnow().isoformat()
                }
            }
            
            if name:
                customer_data["name"] = name
            
            customer = stripe.Customer.create(**customer_data)
            
            logger.info(f"Created Stripe customer for user {user_id}")
            
            return {
                "success": True,
                "customer_id": customer.id,
                "customer": customer
            }
            
        except Exception as e:
            logger.error(f"Error creating Stripe customer: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_subscription(self, customer_id: str, plan_type: str, payment_method_id: str = None) -> Dict[str, Any]:
        """Create a subscription for a customer"""
        try:
            plan = self.plans.get(plan_type)
            if not plan:
                return {
                    "success": False,
                    "error": f"Invalid plan type: {plan_type}"
                }
            
            # Create or get Stripe price
            if plan.stripe_price_id:
                price_id = plan.stripe_price_id
            else:
                # Create price dynamically
                price = stripe.Price.create(
                    currency=plan.currency,
                    unit_amount=int(plan.price * 100),  # Convert to cents
                    recurring={"interval": plan.interval},
                    product_data={
                        "name": plan.name,
                        "description": f"{plan.name} subscription"
                    }
                )
                price_id = price.id
                plan.stripe_price_id = price_id
            
            subscription_data = {
                "customer": customer_id,
                "items": [{"price": price_id}],
                "metadata": {
                    "plan_type": plan_type,
                    "created_at": datetime.utcnow().isoformat()
                }
            }
            
            if payment_method_id:
                subscription_data["default_payment_method"] = payment_method_id
            
            subscription = stripe.Subscription.create(**subscription_data)
            
            logger.info(f"Created subscription for customer {customer_id}")
            
            return {
                "success": True,
                "subscription_id": subscription.id,
                "subscription": subscription,
                "plan": plan
            }
            
        except Exception as e:
            logger.error(f"Error creating subscription: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def cancel_subscription(self, subscription_id: str, immediate: bool = False) -> Dict[str, Any]:
        """Cancel a subscription"""
        try:
            if immediate:
                subscription = stripe.Subscription.delete(subscription_id)
            else:
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
            
            logger.info(f"Cancelled subscription {subscription_id}")
            
            return {
                "success": True,
                "subscription": subscription
            }
            
        except Exception as e:
            logger.error(f"Error cancelling subscription: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_payment_intent(self, amount: float, currency: str = "usd", metadata: Dict[str, str] = None) -> Dict[str, Any]:
        """Create a payment intent for one-time payments"""
        try:
            payment_intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Convert to cents
                currency=currency,
                metadata=metadata or {},
                automatic_payment_methods={"enabled": True}
            )
            
            logger.info(f"Created payment intent for ${amount}")
            
            return {
                "success": True,
                "payment_intent_id": payment_intent.id,
                "client_secret": payment_intent.client_secret,
                "payment_intent": payment_intent
            }
            
        except Exception as e:
            logger.error(f"Error creating payment intent: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def confirm_payment(self, payment_intent_id: str) -> Dict[str, Any]:
        """Confirm a payment intent"""
        try:
            payment_intent = stripe.PaymentIntent.confirm(payment_intent_id)
            
            logger.info(f"Confirmed payment {payment_intent_id}")
            
            return {
                "success": True,
                "payment_intent": payment_intent
            }
            
        except Exception as e:
            logger.error(f"Error confirming payment: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def refund_payment(self, payment_intent_id: str, amount: float = None) -> Dict[str, Any]:
        """Refund a payment"""
        try:
            refund_data = {"payment_intent": payment_intent_id}
            
            if amount:
                refund_data["amount"] = int(amount * 100)  # Convert to cents
            
            refund = stripe.Refund.create(**refund_data)
            
            logger.info(f"Refunded payment {payment_intent_id}")
            
            return {
                "success": True,
                "refund": refund
            }
            
        except Exception as e:
            logger.error(f"Error refunding payment: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_customer_subscriptions(self, customer_id: str) -> Dict[str, Any]:
        """Get all subscriptions for a customer"""
        try:
            subscriptions = stripe.Subscription.list(customer=customer_id)
            
            return {
                "success": True,
                "subscriptions": subscriptions.data
            }
            
        except Exception as e:
            logger.error(f"Error getting customer subscriptions: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_subscription_details(self, subscription_id: str) -> Dict[str, Any]:
        """Get subscription details"""
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            return {
                "success": True,
                "subscription": subscription
            }
            
        except Exception as e:
            logger.error(f"Error getting subscription details: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def update_subscription(self, subscription_id: str, plan_type: str) -> Dict[str, Any]:
        """Update a subscription to a different plan"""
        try:
            plan = self.plans.get(plan_type)
            if not plan:
                return {
                    "success": False,
                    "error": f"Invalid plan type: {plan_type}"
                }
            
            # Get or create price
            if plan.stripe_price_id:
                price_id = plan.stripe_price_id
            else:
                price = stripe.Price.create(
                    currency=plan.currency,
                    unit_amount=int(plan.price * 100),
                    recurring={"interval": plan.interval},
                    product_data={
                        "name": plan.name,
                        "description": f"{plan.name} subscription"
                    }
                )
                price_id = price.id
                plan.stripe_price_id = price_id
            
            subscription = stripe.Subscription.modify(
                subscription_id,
                items=[{"price": price_id}],
                metadata={
                    "plan_type": plan_type,
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Updated subscription {subscription_id} to {plan_type}")
            
            return {
                "success": True,
                "subscription": subscription,
                "plan": plan
            }
            
        except Exception as e:
            logger.error(f"Error updating subscription: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_webhook(self, payload: str, sig_header: str) -> Dict[str, Any]:
        """Process Stripe webhook events"""
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret
            )
            
            event_data = {
                "success": True,
                "event_type": event.type,
                "event_data": event.data
            }
            
            # Handle specific events
            if event.type == "invoice.payment_succeeded":
                await self._handle_payment_succeeded(event.data)
            elif event.type == "invoice.payment_failed":
                await self._handle_payment_failed(event.data)
            elif event.type == "customer.subscription.created":
                await self._handle_subscription_created(event.data)
            elif event.type == "customer.subscription.deleted":
                await self._handle_subscription_deleted(event.data)
            elif event.type == "customer.subscription.updated":
                await self._handle_subscription_updated(event.data)
            
            logger.info(f"Processed webhook event: {event.type}")
            
            return event_data
            
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_payment_succeeded(self, event_data: Dict[str, Any]):
        """Handle successful payment event"""
        # Update user subscription status
        # Send confirmation email
        # Update database records
        pass
    
    async def _handle_payment_failed(self, event_data: Dict[str, Any]):
        """Handle failed payment event"""
        # Notify user of payment failure
        # Update subscription status
        # Send retry notification
        pass
    
    async def _handle_subscription_created(self, event_data: Dict[str, Any]):
        """Handle subscription created event"""
        # Activate user features
        # Send welcome email
        # Update database
        pass
    
    async def _handle_subscription_deleted(self, event_data: Dict[str, Any]):
        """Handle subscription deleted event"""
        # Deactivate user features
        # Send cancellation confirmation
        # Update database
        pass
    
    async def _handle_subscription_updated(self, event_data: Dict[str, Any]):
        """Handle subscription updated event"""
        # Update user features
        # Send confirmation email
        # Update database
        pass
    
    def get_plan_details(self, plan_type: str) -> Optional[SubscriptionPlan]:
        """Get plan details"""
        return self.plans.get(plan_type)
    
    def get_all_plans(self) -> Dict[str, SubscriptionPlan]:
        """Get all available plans"""
        return self.plans
    
    def get_usage_stats(self, customer_id: str) -> Dict[str, Any]:
        """Get usage statistics for a customer"""
        # This would integrate with your usage tracking system
        return {
            "trades_today": 0,
            "trades_this_month": 0,
            "api_calls_today": 0,
            "api_calls_this_month": 0,
            "storage_used": 0,
            "last_activity": datetime.utcnow().isoformat()
        }

# Factory function
def create_stripe_service(api_key: str, webhook_secret: str = None) -> StripeService:
    """Create and return Stripe service instance"""
    return StripeService(api_key, webhook_secret)
