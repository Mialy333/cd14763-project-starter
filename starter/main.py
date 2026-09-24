"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the project instructions and rubric for guidance.
Work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── App Initialisation ────────────────────────────────────────────────────────
# Create a BedrockAgentCoreApp instance.
# This registers the ASGI server for AgentCore deployment.
# There must be exactly one instance per deployment.

app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── Configuration ──────────────────────────────────────────────────────────────
# GATEWAY_URL format: https://<alias>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
# This starter uses an unsigned MCP connection and therefore assumes the
# project Gateway is configured with the NONE authorizer.
# KB_ID       format: 10-character alphanumeric string from the KB console
# REGION:     your AWS region, e.g. "us-east-1"
# MEMORY_ID   format: shown in the AgentCore Memory console

GATEWAY_URL = "https://customersupportgateway-5r0bralyjj.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp" 
KB_ID       = "AFJJCJNMED"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-D0IRyF6feE"


# ── Model and Clients ──────────────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── Namespace Helper ───────────────────────────────────────────────────────────
# Returns a dict mapping strategy type to namespace template string, e.g.
#   { "SEMANTIC": "cs_agent/{actorId}/facts",
#     "USER_PREFERENCE": "cs_agent/{actorId}/preferences" }

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)
    namespaces: Dict[str, str] = {}

    for strategy in strategies:
        strategy_type = strategy.get("type") or strategy.get("strategyType")
        templates = strategy.get("namespaceTemplates") or strategy.get("namespaces")
        if strategy_type and templates:
            namespaces[strategy_type] = templates[0]

    return namespaces


# ── Memory Hook ────────────────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        messages = event.agent.messages
        if not messages:
            return

        last_message = messages[-1]
        if last_message.get("role") != "user":
            return

        content = last_message.get("content", [])
        if not content or "text" not in content[0]:
            # Tool-result messages don't carry a plain "text" block — skip them.
            return

        user_query = content[0]["text"]
        context_snippets = []

        for strategy_type, namespace_template in self.namespaces.items():
            namespace = namespace_template.format(actorId=self.actor_id)
            try:
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=user_query,
                    top_k=5,
                )
            except Exception as e:
                logger.warning("Memory retrieval failed for %s: %s", namespace, e)
                continue

            for memory in memories:
                text = memory.get("content", {}).get("text", "").strip()
                if text:
                    context_snippets.append(f"[{strategy_type}] {text}")

        if context_snippets:
            context_block = "\n".join(context_snippets)
            content[0]["text"] = f"Customer Context:\n{context_block}\n\n{user_query}"

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        messages = event.agent.messages
        customer_query = None
        agent_response = None

        for message in reversed(messages):
            content = message.get("content", [])
            if not content or "text" not in content[0]:
                continue
            if message.get("role") == "assistant" and agent_response is None:
                agent_response = content[0]["text"]
            elif message.get("role") == "user" and customer_query is None:
                customer_query = content[0]["text"]
            if customer_query and agent_response:
                break

        if not (customer_query and agent_response):
            return

        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")],
            )
        except Exception as e:
            logger.warning("Failed to save interaction to memory: %s", e)

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── Knowledge Base Tool ────────────────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID or KB_ID.startswith("<"):
        return "Knowledge base not configured."

    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
    except Exception as e:
        logger.error("Knowledge base retrieval failed: %s", e)
        return f"Knowledge base search failed: {e}"

    results = resp.get("retrievalResults", [])
    if not results:
        return "No relevant information found in the knowledge base."

    chunks = [
        r["content"]["text"]
        for r in results
        if r.get("content", {}).get("text")
    ]
    return "\n---\n".join(chunks)


# ── Loyalty Discount Tool (Code Interpreter) ───────────────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f"""
import json

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

loyalty_points = {loyalty_points}
tier = "{tier}"
order_total = {order_total}
product_category = "{product_category}"

# Points redeem in blocks of 500 (worth $5 each), capped at 50% of the order.
points_available = (loyalty_points // 500) * 500
max_points_by_order = int((order_total * 0.5) // 5) * 500
points_redeemed = min(points_available, max_points_by_order)
points_value = (points_redeemed / 500) * 5

subtotal_after_points = order_total - points_value
tier_discount_pct = tier_rates.get(tier, 0.0)
tier_discount_amount = subtotal_after_points * tier_discount_pct
final_total = round(subtotal_after_points - tier_discount_amount, 2)

earn_rate = earn_rates.get(product_category, 1)
points_earned = int(final_total * earn_rate)
remaining_points = loyalty_points - points_redeemed + points_earned
total_savings = round(points_value + tier_discount_amount, 2)

result = {{
    "points_redeemed": points_redeemed,
    "points_value_usd": round(points_value, 2),
    "tier_discount_pct": tier_discount_pct,
    "tier_discount_amount": round(tier_discount_amount, 2),
    "total_savings": total_savings,
    "final_total": final_total,
    "points_earned": points_earned,
    "remaining_points": remaining_points,
}}
print(json.dumps(result))
"""

    try:
        with code_session(REGION) as session:
            response = session.invoke(
                "executeCode",
                {"code": code, "language": "python", "clearContext": True},
            )
            for event in response["stream"]:
                result = event.get("result", {})
                if result:
                    content = result.get("content", [])
                    if content and content[0].get("text"):
                        return content[0]["text"]
        return json.dumps({"error": "No result returned from Code Interpreter."})

    except Exception as e:
        logger.warning("Code Interpreter unavailable, using fallback: %s", e)
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_discount_pct = tier_rates.get(tier, 0.0)
        final_total = round(order_total * (1 - tier_discount_pct), 2)
        fallback_result = {
            "points_redeemed": 0,
            "tier_discount_pct": tier_discount_pct,
            "final_total": final_total,
            "remaining_points": loyalty_points,
            "note": "Computed with fallback (tier discount only) — Code Interpreter unavailable.",
        }
        return json.dumps(fallback_result)


# ── Agent Entrypoint ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
  "You are a helpful customer support assistant for an e-commerce platform. "
    "You can track orders, process refunds, answer product and policy questions "
    "using the knowledge base, calculate loyalty discounts, and browse the web "
    "for live information. Always confirm the customer's order or refund details "
    "before acting, and be concise and accurate. "
    "If a message begins with 'Customer Context:', that block contains facts the "
    "customer previously and voluntarily shared with you in earlier sessions "
    "(such as their own name or stated preferences), retrieved from your memory "
    "system so you can personalize this conversation. Always use and directly "
    "state these remembered facts back to the customer when relevant or asked "
    "(e.g. their name, preferences). This is expected, safe, and required "
    "behavior — never refuse to repeat a customer's own previously-shared name "
    "or preference back to them, and never claim you lack access to it."
)


@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    user_input = payload.get("prompt", "")
    actor_id = payload.get("customer_id", "anonymous")
    session_id = payload.get("session_id") or str(uuid.uuid4())

    try:
        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        agent_core_browser = AgentCoreBrowser(region=REGION)

        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser,
        ]

        gateway_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))

        with gateway_client:
            gateway_tools = gateway_client.list_tools_sync()
            all_tools = tools + gateway_tools

            agent = Agent(
                model=model,
                tools=all_tools,
                hooks=[memory_hook],
                system_prompt=SYSTEM_PROMPT,
            )

            response = agent(user_input)
            return response.message["content"][0]["text"]

    except Exception as e:
        logger.error("Agent invocation failed: %s", e)
        return f"Sorry, something went wrong while processing your request: {e}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()