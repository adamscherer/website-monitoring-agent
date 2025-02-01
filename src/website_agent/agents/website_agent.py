import logging
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import yaml
import schedule
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("WebsiteAgent")


class WebsiteAgent:
    """Agent responsible for monitoring website status and performance with ML-based anomaly detection."""

    def __init__(self, config_path="config/config.yaml"):
        self.logger = logger
        self.config = self._load_config(config_path)
        self.performance_history = []
        self.baseline_metrics = None
        self.current_check_interval = self.config["website"]["check_interval"]
        self.learning_cycles = 0
        
        # ML components
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expected proportion of anomalies
            random_state=42,
            n_estimators=100
        )
        self.ml_ready = False
        self.min_samples_for_ml = 50  # Minimum samples needed before ML kicks in

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

    def _prepare_ml_data(self):
        """Prepare data for machine learning analysis"""
        if len(self.performance_history) < self.min_samples_for_ml:
            return None
            
        # Extract numerical features
        data = []
        for metrics in self.performance_history:
            features = {
                'response_time': float(metrics['response_time'].replace('s', '')),
                'content_length': metrics['content_length'],
                'links_count': metrics['links_count'],
                'images_count': metrics['images_count'],
                'status_code': metrics['status_code']
            }
            data.append(features)
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df

    def _train_ml_model(self):
        """Train the anomaly detection model"""
        df = self._prepare_ml_data()
        if df is None:
            return False
            
        # Standardize the features
        scaled_data = self.scaler.fit_transform(df)
        
        # Train the model
        self.isolation_forest.fit(scaled_data)
        self.ml_ready = True
        self.logger.info("ML model trained successfully")
        return True

    def _detect_anomalies_ml(self, current_metrics):
        """Detect anomalies using machine learning"""
        if not self.ml_ready:
            return []
            
        # Prepare current metrics
        current_data = pd.DataFrame([{
            'response_time': float(current_metrics['response_time'].replace('s', '')),
            'content_length': current_metrics['content_length'],
            'links_count': current_metrics['links_count'],
            'images_count': current_metrics['images_count'],
            'status_code': current_metrics['status_code']
        }])
        
        # Scale the current data
        scaled_current = self.scaler.transform(current_data)
        
        # Predict anomaly (-1 for anomaly, 1 for normal)
        prediction = self.isolation_forest.predict(scaled_current)
        
        # Get anomaly score (negative score indicates anomaly)
        anomaly_score = self.isolation_forest.score_samples(scaled_current)[0]
        
        decisions = []
        if prediction[0] == -1:
            # This is an anomaly
            severity = "high" if anomaly_score < -0.5 else "medium"
            decisions.append({
                "type": "alert",
                "action": "ml_anomaly",
                "reason": f"ML model detected {severity} severity anomaly (score: {anomaly_score:.3f})",
                "severity": severity,
                "anomaly_score": anomaly_score
            })
            
            # Adjust monitoring frequency for high-severity anomalies
            if severity == "high":
                decisions.append({
                    "type": "adjust_monitoring",
                    "action": "decrease_interval",
                    "reason": "High-severity anomaly detected by ML model"
                })
                
        return decisions

    def analyze_metrics(self, metrics):
        """Analyze metrics and make autonomous decisions using both traditional and ML approaches"""
        # Update performance history
        self.performance_history.append(metrics)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
            
        # Train/update ML model every 10 cycles if we have enough data
        if len(self.performance_history) >= self.min_samples_for_ml and self.learning_cycles % 10 == 0:
            self._train_ml_model()
            
        self.learning_cycles += 1
        
        # Get decisions from both traditional and ML approaches
        traditional_decisions = self._make_decisions(metrics)
        ml_decisions = self._detect_anomalies_ml(metrics)
        
        # Combine and execute all decisions
        all_decisions = traditional_decisions + ml_decisions
        self._execute_decisions(all_decisions)

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

    def _execute_decisions(self, decisions):
        """Execute the decisions made by the agent"""
        for decision in decisions:
            self.logger.info(f"Executing decision: {decision}")
            
            if decision["type"] == "adjust_monitoring":
                if decision["action"] == "decrease_interval":
                    new_interval = max(0.5, self.current_check_interval * 0.5)
                    self.current_check_interval = new_interval
                    schedule.clear()
                    schedule.every(new_interval).minutes.do(self.monitor_website)
                    self.logger.info(f"Adjusted monitoring interval to {new_interval} minutes")
                    
            elif decision["type"] == "alert":
                if decision["action"] == "critical_error":
                    self.logger.critical(f"CRITICAL ALERT: {decision['reason']}")
                elif decision["action"] == "warning":
                    self.logger.warning(f"WARNING: {decision['reason']}")
                elif decision["action"] == "content_change":
                    self.logger.warning(f"CONTENT ALERT: {decision['reason']}")
                elif decision["action"] == "ml_anomaly":
                    severity = decision.get("severity", "unknown")
                    self.logger.warning(
                        f"ML ANOMALY ({severity.upper()}): {decision['reason']}\n"
                        f"Anomaly Score: {decision.get('anomaly_score', 'N/A')}"
                    )

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
