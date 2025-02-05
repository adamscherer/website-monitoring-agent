import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.website_agent.agents.langchain_website_agent import WebsiteMonitoringAgent

# Set up logging
logging.basicConfig(level=logging.INFO)


async def main():
    # Load environment variables
    load_dotenv()

    # Initialize the agent
    agent = WebsiteMonitoringAgent(
        config_path="config/config.yaml", openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Monitor continuously
    while True:
        try:
            result = await agent.monitor_website()
            logging.info(f"Monitoring result: {result}")

            # Wait for the configured interval
            await asyncio.sleep(agent.config["website"]["check_interval"] * 60)
        except Exception as e:
            logging.error(f"Error during monitoring: {str(e)}")
            await asyncio.sleep(60)  # Wait a minute before retrying


if __name__ == "__main__":
    asyncio.run(main())
