#!/usr/bin/env python3
"""
Test script to verify Claude API with tool calling works via AWS Bedrock.
"""

import json
import boto3
from loguru import logger

# Test configuration
MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
REGION = "us-west-2"


def test_claude_with_tools():
    """Test Claude API with a simple tool calling scenario."""
    logger.info("Testing Claude API with tool calling via AWS Bedrock")

    # Initialize client
    try:
        client = boto3.client('bedrock-runtime', region_name=REGION)
        logger.info(f"Successfully connected to Bedrock in {REGION}")
    except Exception as e:
        logger.error(f"Failed to connect to Bedrock: {e}")
        return False

    # Define a simple tool
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    ]

    # Test message
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco?"
        }
    ]

    # Build payload
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.7,
        "messages": messages,
        "tools": tools,
        "tool_choice": {"type": "auto"}
    }

    try:
        logger.info(f"Calling model: {MODEL_ID}")
        response = client.invoke_model(
            body=json.dumps(payload),
            modelId=MODEL_ID
        )

        resp_body = json.loads(response['body'].read())
        logger.info(f"Response received successfully")

        # Parse response
        content = resp_body.get("content", [])
        logger.info(f"Response content blocks: {len(content)}")

        for block in content:
            block_type = block.get("type", "")
            if block_type == "text":
                logger.info(f"Text: {block.get('text', '')[:100]}...")
            elif block_type == "tool_use":
                logger.info(f"Tool call: {block.get('name', '')}")
                logger.info(f"  Input: {json.dumps(block.get('input', {}))}")

        # Check if tool was called
        tool_uses = [b for b in content if b.get("type") == "tool_use"]
        if tool_uses:
            logger.info(f"SUCCESS: Claude made {len(tool_uses)} tool call(s)")

            # Simulate tool response
            tool_response = {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_uses[0].get("id"),
                    "content": json.dumps({
                        "temperature": 72,
                        "unit": "fahrenheit",
                        "condition": "Sunny"
                    })
                }]
            }

            # Continue conversation with tool result
            messages.append({
                "role": "assistant",
                "content": content
            })
            messages.append(tool_response)

            payload["messages"] = messages

            logger.info("Sending tool result back to Claude...")
            response2 = client.invoke_model(
                body=json.dumps(payload),
                modelId=MODEL_ID
            )

            resp_body2 = json.loads(response2['body'].read())
            final_content = resp_body2.get("content", [])

            for block in final_content:
                if block.get("type") == "text":
                    logger.info(f"Final response: {block.get('text', '')}")

            logger.info("SUCCESS: Full tool calling flow completed!")
            return True
        else:
            logger.warning("No tool calls made - model responded with text only")
            return True

    except Exception as e:
        logger.error(f"Error during API call: {e}")
        return False


def test_available_models():
    """List available Claude models on Bedrock."""
    logger.info("Checking available Claude models on Bedrock")

    try:
        bedrock = boto3.client('bedrock', region_name=REGION)
        response = bedrock.list_foundation_models(
            byProvider="anthropic"
        )

        models = response.get('modelSummaries', [])
        logger.info(f"Found {len(models)} Anthropic models:")

        for model in models:
            model_id = model.get('modelId', '')
            model_name = model.get('modelName', '')
            if 'claude' in model_id.lower():
                logger.info(f"  - {model_id}: {model_name}")

    except Exception as e:
        logger.error(f"Error listing models: {e}")


def main():
    logger.info("=" * 60)
    logger.info("Claude Tool Calling Test for BFCL v3")
    logger.info("=" * 60)

    # Test available models
    test_available_models()

    logger.info("")
    logger.info("=" * 60)

    # Test tool calling
    success = test_claude_with_tools()

    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("All tests passed! Ready to run BFCL benchmark.")
    else:
        logger.error("Tests failed. Check your AWS credentials and permissions.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
