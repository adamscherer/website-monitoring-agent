from typing import Dict, List, Any, Optional, Annotated, TypedDict, Tuple
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import yaml
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger("LangChainWebsiteAgent")

# Define state types for the agent
class AgentState(TypedDict):
    messages: List[BaseMessage]
    metrics: Dict[str, Any]
    performance_history: List[Dict[str, Any]]
    baseline_metrics: Optional[Dict[str, Any]]
    ml_model: Optional[Dict[str, Any]]

class WebsiteMetrics(BaseModel):
    """Model for website monitoring metrics"""
    status_code: int = Field(..., description="HTTP status code from the response")
    response_time: float = Field(..., description="Response time in seconds")
    content_length: int = Field(..., description="Length of the response content in bytes")
    links_count: int = Field(..., description="Number of links found on the page")
    images_count: int = Field(..., description="Number of images found on the page")
    timestamp: datetime = Field(default_factory=datetime.now)

class MLAnomalyResult(BaseModel):
    """Model for ML-based anomaly detection results"""
    is_anomaly: bool = Field(..., description="Whether the current metrics are anomalous")
    anomaly_score: float = Field(..., description="Anomaly score from the ML model")
    severity: str = Field(..., description="Severity of the anomaly: low, medium, or high")
    confidence: float = Field(..., description="Confidence in the anomaly detection")

class WebsiteMonitoringAgent:
    """LangChain-based website monitoring agent with ML capabilities"""

    def __init__(self, config_path: str = "config/config.yaml", openai_api_key: str = None):
        self.config = self._load_config(config_path)
        self.performance_history: List[Dict[str, Any]] = []
        self.baseline_metrics: Optional[Dict[str, Any]] = None
        
        # Initialize ML components
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.ml_ready = False
        self.min_samples_for_ml = 50
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0.0,
            openai_api_key=openai_api_key
        )
        
        # Create the agent graph
        self.graph = self._create_agent_graph()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return self._create_default_config()
            
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            'website': {
                'url': 'http://example.com',
                'check_interval': 1  # minutes
            },
            'monitoring': {
                'enabled': True,
                'check_uptime': True,
                'check_performance': True
            }
        }

    @tool("collect_metrics")
    def collect_metrics(self, url: str) -> Dict[str, Any]:
        """Collect metrics from the specified website"""
        try:
            start_time = datetime.now()
            response = requests.get(url)
            response_time = (datetime.now() - start_time).total_seconds()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            metrics = {
                'status_code': response.status_code,
                'response_time': response_time,
                'content_length': len(response.content),
                'links_count': len(soup.find_all('a')),
                'images_count': len(soup.find_all('img')),
                'timestamp': datetime.now().isoformat()
            }
            return metrics
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            raise

    @tool("detect_anomalies")
    def detect_anomalies(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in the current metrics using ML"""
        if not self.ml_ready:
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'severity': 'unknown',
                'confidence': 0.0
            }
            
        current_data = np.array([[
            float(metrics['response_time']),
            int(metrics['content_length']),
            int(metrics['links_count']),
            int(metrics['images_count']),
            int(metrics['status_code'])
        ]])
        
        scaled_data = self.scaler.transform(current_data)
        anomaly_score = float(self.isolation_forest.score_samples(scaled_data)[0])
        
        is_anomaly = anomaly_score < -0.5
        severity = (
            "high" if anomaly_score < -0.7
            else "medium" if anomaly_score < -0.5
            else "low"
        )
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': float(anomaly_score),
            'severity': severity,
            'confidence': 0.8 if self.learning_cycles > 100 else 0.5
        }

    @tool("update_baseline")
    def update_baseline(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Update baseline metrics with new data"""
        self.performance_history.append(metrics)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
            
        if len(self.performance_history) >= self.min_samples_for_ml:
            # Convert history to numpy array for ML
            data = np.array([[
                float(m['response_time']),
                int(m['content_length']),
                int(m['links_count']),
                int(m['images_count']),
                int(m['status_code'])
            ] for m in self.performance_history])
            
            # Update ML model
            self.scaler.fit(data)
            self.isolation_forest.fit(self.scaler.transform(data))
            self.ml_ready = True
            
            # Update baseline metrics
            self.baseline_metrics = {
                'avg_response_time': float(np.mean([float(m['response_time']) for m in self.performance_history])),
                'std_response_time': float(np.std([float(m['response_time']) for m in self.performance_history])),
                'avg_content_length': float(np.mean([int(m['content_length']) for m in self.performance_history])),
                'typical_status_code': int(np.median([int(m['status_code']) for m in self.performance_history]))
            }
            
        return self.baseline_metrics or {}

    def _create_agent_graph(self) -> StateGraph:
        """Create the agent's workflow graph"""
        from langgraph.prebuilt import ToolExecutor
        from typing import Annotated, TypedDict, Union, Literal
        
        # Define state schema
        class MonitorState(TypedDict):
            config: Dict[str, str]
            messages: List[Dict[str, Any]]
            metrics: Dict[str, Any]
            analysis: Dict[str, Any]
            baseline: Dict[str, Any]
            performance_history: List[Dict[str, Any]]
            ml_model: Dict[str, Any]
        
        # Create tool executor
        tools = [
            self.collect_metrics,
            self.detect_anomalies,
            self.update_baseline
        ]
        tool_executor = ToolExecutor(tools)
        
        # Define the nodes
        def collect_metrics_node(state: MonitorState):
            url = state["config"]["url"]
            metrics = tool_executor.invoke("collect_metrics", {"url": url})
            state["metrics"] = metrics
            return {"state": state, "next": "analyze_metrics"}
            
        def analyze_metrics_node(state: MonitorState):
            metrics = state["metrics"]
            analysis = tool_executor.invoke("detect_anomalies", {"metrics": metrics})
            state["analysis"] = analysis
            return {"state": state, "next": "update_baseline"}
            
        def update_baseline_node(state: MonitorState):
            metrics = state["metrics"]
            baseline = tool_executor.invoke("update_baseline", {"metrics": metrics})
            state["baseline"] = baseline
            return {"state": state, "next": None}
        
        # Create the graph
        workflow = StateGraph(MonitorState)
        
        # Add nodes
        workflow.add_node("collect_metrics", collect_metrics_node)
        workflow.add_node("analyze_metrics", analyze_metrics_node)
        workflow.add_node("update_baseline", update_baseline_node)
        
        # Add edges
        workflow.add_edge("collect_metrics", "analyze_metrics")
        workflow.add_edge("analyze_metrics", "update_baseline")
        workflow.add_edge("update_baseline", END)
        
        # Set entry point
        workflow.set_entry_point("collect_metrics")
        
        # Compile the graph
        self.app = workflow.compile()
        
        return workflow

    async def monitor_website(self) -> Dict[str, Any]:
        """Monitor website and return results"""
        # Initialize state
        state = {
            "messages": [],
            "metrics": {},
            "performance_history": self.performance_history,
            "baseline_metrics": self.baseline_metrics,
            "ml_model": {
                'ready': self.ml_ready,
                'min_samples': self.min_samples_for_ml
            },
            "config": {"url": self.config['website']['url']}
        }
        
        # Run the workflow
        result = await self.app.ainvoke(state)
        
        return result
