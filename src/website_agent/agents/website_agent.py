import logging
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import yaml
import schedule

logger = logging.getLogger("WebsiteAgent")


class WebsiteAgent:
    """Agent responsible for monitoring website status and performance."""

    def __init__(self, config_path="config/config.yaml"):
        self.logger = logger
        self.config = self._load_config(config_path)
        self.performance_history = []
        self.baseline_metrics = None
        self.current_check_interval = self.config["website"]["check_interval"]
        self.learning_cycles = 0

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(
                f"Config file {config_path} not found. Using default configuration."
            )
            return self._create_default_config()

    def _create_default_config(self):
        """Create default configuration"""
        return {
            "website": {"url": "http://example.com", "check_interval": 5},  # minutes
            "monitoring": {"enabled": True, "check_uptime": True, "check_performance": True},
        }

    def analyze_metrics(self, metrics):
        """Analyze metrics and make autonomous decisions"""
        # Update performance history
        self.performance_history.append(metrics)
        if len(self.performance_history) > 100:  # Keep last 100 records
            self.performance_history.pop(0)

        # Learn and update baseline after collecting enough data
        if len(self.performance_history) >= 10 and self.learning_cycles % 10 == 0:
            self._update_baseline_metrics()

        self.learning_cycles += 1

        # Make decisions based on current metrics
        decisions = self._make_decisions(metrics)
        self._execute_decisions(decisions)

    def _update_baseline_metrics(self):
        """Update baseline metrics based on historical data"""
        if not self.performance_history:
            return

        # Calculate average metrics
        response_times = [
            float(m["response_time"].replace("s", "")) for m in self.performance_history
        ]
        content_lengths = [m["content_length"] for m in self.performance_history]

        self.baseline_metrics = {
            "avg_response_time": sum(response_times) / len(response_times),
            "std_response_time": (
                sum((x - (sum(response_times) / len(response_times))) ** 2 for x in response_times)
                / len(response_times)
            )
            ** 0.5,
            "avg_content_length": sum(content_lengths) / len(content_lengths),
            "typical_status_code": max(
                set(m["status_code"] for m in self.performance_history),
                key=self.performance_history.count,
            ),
        }

        self.logger.info(f"Updated baseline metrics: {self.baseline_metrics}")

    def _make_decisions(self, metrics):
        """Make autonomous decisions based on current metrics and baseline"""
        decisions = []

        # Convert response time to float
        current_response_time = float(metrics["response_time"].replace("s", ""))

        # Check for significant deviations if baseline exists
        if self.baseline_metrics:
            # Response time anomaly detection
            if (
                current_response_time
                > self.baseline_metrics["avg_response_time"]
                + 2 * self.baseline_metrics["std_response_time"]
            ):
                decisions.append(
                    {
                        "type": "adjust_monitoring",
                        "action": "decrease_interval",
                        "reason": "Response time significantly above baseline",
                    }
                )

            # Content length anomaly detection
            content_diff_percent = (
                abs(metrics["content_length"] - self.baseline_metrics["avg_content_length"])
                / self.baseline_metrics["avg_content_length"]
                * 100
            )
            if content_diff_percent > 20:  # More than 20% difference
                decisions.append(
                    {
                        "type": "alert",
                        "action": "content_change",
                        "reason": f"Content length changed by {content_diff_percent:.1f}%",
                    }
                )

        # Status code based decisions
        if metrics["status_code"] >= 500:
            decisions.append(
                {
                    "type": "alert",
                    "action": "critical_error",
                    "reason": f'Server error detected (status code: {metrics["status_code"]})',
                }
            )
            decisions.append(
                {
                    "type": "adjust_monitoring",
                    "action": "decrease_interval",
                    "reason": "Server errors detected",
                }
            )
        elif metrics["status_code"] >= 400:
            decisions.append(
                {
                    "type": "alert",
                    "action": "warning",
                    "reason": f'Client error detected (status code: {metrics["status_code"]})',
                }
            )

        return decisions

    def _execute_decisions(self, decisions):
        """Execute the decisions made by the agent"""
        for decision in decisions:
            self.logger.info(f"Executing decision: {decision}")

            if decision["type"] == "adjust_monitoring":
                if decision["action"] == "decrease_interval":
                    new_interval = max(
                        0.5, self.current_check_interval * 0.5
                    )  # Don't go below 30 seconds
                    self.current_check_interval = new_interval
                    schedule.clear()
                    schedule.every(new_interval).minutes.do(self.monitor_website)
                    self.logger.info(f"Adjusted monitoring interval to {new_interval} minutes")

            elif decision["type"] == "alert":
                # Here you could implement different alert mechanisms (email, Slack, etc.)
                if decision["action"] == "critical_error":
                    self.logger.critical(f"CRITICAL ALERT: {decision['reason']}")
                elif decision["action"] == "warning":
                    self.logger.warning(f"WARNING: {decision['reason']}")
                elif decision["action"] == "content_change":
                    self.logger.warning(f"CONTENT ALERT: {decision['reason']}")

    def monitor_website(self):
        """Monitor website status and performance"""
        url = self.config["website"]["url"]
        self.logger.info(f"Checking website: {url}")

        try:
            start_time = time.time()
            response = requests.get(url)
            response_time = time.time() - start_time

            soup = BeautifulSoup(response.text, "html.parser")

            metrics = {
                "status_code": response.status_code,
                "response_time": f"{response_time:.2f}s",
                "content_length": len(response.content),
                "title": soup.title.string if soup.title else "No title",
                "links_count": len(soup.find_all("a")),
                "images_count": len(soup.find_all("img")),
                "timestamp": datetime.now().isoformat(),
            }

            # Log the results
            self.logger.info("Website check completed successfully:")
            for key, value in metrics.items():
                self.logger.info(f"  {key}: {value}")

            # Analyze metrics and make decisions
            self.analyze_metrics(metrics)

        except requests.RequestException as e:
            self.logger.error(f"Error monitoring website: {str(e)}")
            return None

    def run(self):
        """Start the agent"""
        self.logger.info(
            f"Starting Website Management Agent with config: {self.config['website']['check_interval']}"
        )

        # Schedule regular tasks
        schedule.every(self.config["website"]["check_interval"]).minutes.do(self.monitor_website)

        # Run continuously
        while True:
            schedule.run_pending()
            time.sleep(1)
