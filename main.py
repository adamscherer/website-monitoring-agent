import logging
from dotenv import load_dotenv
from src.website_agent.agents.website_agent import WebsiteAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("agent.log"), logging.StreamHandler()],
)


def main():
    agent = WebsiteAgent()
    agent.run()


if __name__ == "__main__":
    main()
