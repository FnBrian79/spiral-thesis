#!/usr/bin/env python3
"""
Maestro Hybrid AI Orchestrator
A robust console for managing local and cloud AI interactions with full auditability.
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator
import requests


# ==================== Configuration ====================

class Config:
    """Externalized configuration with environment variable support"""

    def __init__(self):
        # Local model settings
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self.default_local_model = os.getenv('LOCAL_MODEL', 'mistral')

        # Cloud settings
        self.cloud_api_key = os.getenv('CLOUD_API_KEY', '')
        self.cloud_endpoint = os.getenv('CLOUD_ENDPOINT', 'https://api.openai.com/v1/chat/completions')
        self.cloud_model = os.getenv('CLOUD_MODEL', 'gpt-4')

        # Notebook settings
        self.notebook_dir = Path(os.getenv('NOTEBOOK_DIR', './notebooks'))
        self.notebook_dir.mkdir(parents=True, exist_ok=True)

        # Timeouts (seconds)
        self.local_timeout = int(os.getenv('LOCAL_TIMEOUT', '30'))
        self.cloud_timeout = int(os.getenv('CLOUD_TIMEOUT', '60'))

    @classmethod
    def from_file(cls, config_path: str):
        """Load config from JSON/YAML file"""
        config = cls()
        if Path(config_path).exists():
            with open(config_path) as f:
                data = json.load(f)
                for key, value in data.items():
                    setattr(config, key, value)
        return config


# ==================== Logging Setup ====================

def setup_logging(verbose: bool = False):
    """Configure logging with verbosity control"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger('maestro')


# ==================== Notebook Manager ====================

class NotebookManager:
    """Handles conversation logging with rich formatting"""

    def __init__(self, config: Config, session_name: Optional[str] = None):
        self.config = config
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session = session_name or f'session_{timestamp}'
        self.notebook_path = config.notebook_dir / f'{session}.md'
        self.exchanges = 0

    def write_header(self, model_info: dict):
        """Write session metadata"""
        with open(self.notebook_path, 'w') as f:
            f.write(f"# Maestro Session\n\n")
            f.write(f"**Started:** {datetime.now().isoformat()}\n\n")
            f.write(f"**Model:** {model_info.get('name', 'Unknown')}\n")
            f.write(f"**Type:** {model_info.get('type', 'Unknown')}\n")
            f.write(f"**Endpoint:** {model_info.get('endpoint', 'N/A')}\n")
            f.write(f"\n---\n\n")

    def log_exchange(self, user_input: str, ai_response: str, model: str, metrics: Optional[dict] = None):
        """Log a user-AI exchange with formatting"""
        self.exchanges += 1
        with open(self.notebook_path, 'a') as f:
            f.write(f"## Exchange {self.exchanges}\n\n")
            f.write(f"**User:**\n{user_input}\n\n")
            f.write(f"**AI ({model}):**\n```\n{ai_response}\n```\n\n")
            if metrics:
                f.write(f"**Metrics:**\n")
                for k, v in metrics.items():
                    f.write(f"- {k}: {v}\n")
                f.write("\n")
            f.write(f"---\n\n")

    def log_system_note(self, note: str, details: Optional[str] = None):
        """Log system events for educational purposes"""
        with open(self.notebook_path, 'a') as f:
            f.write(f"### ℹ️ System Note\n\n")
            f.write(f"{note}\n\n")
            if details:
                f.write(f"```\n{details}\n```\n\n")
            f.write(f"---\n\n")

    def log_error(self, error: Exception, context: str):
        """Log errors with full stack trace for auditability"""
        with open(self.notebook_path, 'a') as f:
            f.write(f"### ⚠️ Error in {context}\n\n")
            f.write(f"**Type:** `{type(error).__name__}`\n\n")
            f.write(f"**Message:** {str(error)}\n\n")
            f.write(f"```python\n")
            import traceback
            f.write(traceback.format_exc())
            f.write(f"```\n\n---\n\n")

    def write_footer(self, summary: dict):
        """Write session summary on graceful exit"""
        with open(self.notebook_path, 'a') as f:
            f.write(f"\n## Session Summary\n\n")
            f.write(f"**Ended:** {datetime.now().isoformat()}\n\n")
            f.write(f"**Total Exchanges:** {self.exchanges}\n")
            f.write(f"**Duration:** {summary.get('duration', 'N/A')}\n")
            f.write(f"**Model Switches:** {summary.get('switches', 0)}\n")


# ==================== Model Providers ====================

class LocalProvider:
    """Handles local Ollama inference"""

    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger
        self.history = []
        self.name = "Local (Ollama)"

    def list_models(self) -> list:
        """Discover available local models with metadata"""
        try:
            response = requests.get(
                f"{self.config.ollama_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            models = response.json().get('models', [])
            return [
                {
                    'name': m['name'],
                    'size': m.get('size', 'Unknown'),
                    'family': m.get('details', {}).get('family', 'Unknown')
                }
                for m in models
            ]
        except Exception as e:
            self.logger.error(f"Failed to list local models: {e}")
            return []

    def chat(self, user_input: str, model: str) -> Generator[str, None, None]:
        """Stream response from local model"""
        self.history.append({"role": "user", "content": user_input})

        payload = {
            "model": model,
            "messages": self.history,
            "stream": True
        }

        try:
            with requests.post(
                f"{self.config.ollama_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.config.local_timeout
            ) as response:
                response.raise_for_status()

                full_response = ""
                for line in response.iter_lines():
                    if line:
                        body = json.loads(line.decode('utf-8'))
                        if 'message' in body:
                            content = body['message'].get('content', '')
                            full_response += content
                            yield content
                        if body.get('done'):
                            self.history.append({
                                "role": "assistant",
                                "content": full_response
                            })
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Local inference failed: {e}")
            raise


class CloudProvider:
    """Handles cloud API inference (OpenAI-compatible)"""

    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger
        self.history = []
        self.name = "Cloud (API)"

    def chat(self, user_input: str, model: str = "gpt-4") -> Generator[str, None, None]:
        """Stream response from cloud API"""
        self.logger.info(f"Cloud inference requested for model: {model}")
        self.history.append({"role": "user", "content": user_input})

        if not self.config.cloud_api_key:
             yield "[Error: CLOUD_API_KEY not configured. Please set it in config or environment variables.]"
             return

        headers = {
            "Authorization": f"Bearer {self.config.cloud_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": self.history,
            "stream": True
        }

        try:
            with requests.post(
                self.config.cloud_endpoint,
                headers=headers,
                json=payload,
                stream=True,
                timeout=self.config.cloud_timeout
            ) as response:
                response.raise_for_status()

                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith("data: "):
                            if line_text == "data: [DONE]":
                                break
                            try:
                                data = json.loads(line_text[6:])
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    full_response += content
                                    yield content
                            except json.JSONDecodeError:
                                continue

                self.history.append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Cloud inference failed: {e}")
            yield f"[Cloud Error: {str(e)}]"
            raise


# ==================== Main Orchestrator ====================

class Maestro:
    """Main orchestration class with hybrid mode support"""

    def __init__(self, config: Config, logger, mode: str = 'hybrid', explain_mode: bool = False):
        self.config = config
        self.logger = logger
        self.mode = mode
        self.explain_mode = explain_mode
        self.notebook = None

        self.local = LocalProvider(config, logger)
        self.cloud = CloudProvider(config, logger)

        self.start_time = datetime.now()
        self.model_switches = 0

    def initialize_session(self, session_name: Optional[str] = None):
        """Initialize notebook and session metadata"""
        self.notebook = NotebookManager(self.config, session_name)

        model_info = {
            'name': self.config.default_local_model if self.mode != 'cloud' else self.config.cloud_model,
            'type': self.mode,
            'endpoint': self.config.ollama_url if self.mode != 'cloud' else self.config.cloud_endpoint
        }

        self.notebook.write_header(model_info)
        self.logger.info(f"Session initialized: {self.notebook.notebook_path}")

    def explain(self, message: str):
        """Print explanatory messages if in explain mode"""
        if self.explain_mode:
            print(f"\n🔍 \033[94m[Maestro Logic]: {message}\033[0m")

    def chat_loop(self):
        """Main interactive chat loop with graceful shutdown"""
        print("\n🎵 Maestro Hybrid AI Console")
        print(f"Mode: {self.mode}")
        print("Commands: /switch, /models, /doctor, /exit\n")

        if self.mode == 'cloud':
            current_model = self.config.cloud_model
            provider = self.cloud
        else:
            current_model = self.config.default_local_model
            provider = self.local

        try:
            while True:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    if user_input == '/exit':
                        break
                    elif user_input == '/switch':
                        provider, current_model = self.handle_switch(provider)
                        continue
                    elif user_input == '/models':
                        self.show_models()
                        continue
                    elif user_input == '/doctor':
                        self.run_doctor()
                        continue

                # Process regular chat
                print("AI: ", end="", flush=True)
                full_response = ""
                start_ts = time.time()

                try:
                    for chunk in provider.chat(user_input, current_model):
                        print(chunk, end="", flush=True)
                        full_response += chunk
                    print()  # Newline after stream

                    duration = time.time() - start_ts
                    metrics = {"latency_sec": f"{duration:.2f}", "provider": provider.name}
                    self.notebook.log_exchange(user_input, full_response, current_model, metrics)
                    self.explain(f"Response received from {provider.name} in {duration:.2f}s.")

                except Exception as e:
                    self.logger.error(f"Inference failed: {e}")
                    self.notebook.log_error(e, f"chat with {current_model}")
                    self.explain(f"Primary provider {provider.name} failed. Error: {e}")

                    # Fallback logic
                    if self.mode == 'hybrid' and provider == self.local:
                        self.logger.info("Falling back to cloud...")
                        self.explain("Initiating fallback to Cloud provider due to local failure.")
                        self.notebook.log_system_note("Fallback triggered", f"Local model failed: {e}. Switching to Cloud.")

                        provider = self.cloud
                        current_model = self.config.cloud_model
                        self.model_switches += 1

                        # Retry with cloud
                        try:
                            print("\nAI (Cloud Fallback): ", end="", flush=True)
                            full_response = ""
                            start_ts = time.time()
                            for chunk in provider.chat(user_input, current_model):
                                print(chunk, end="", flush=True)
                                full_response += chunk
                            print()
                            duration = time.time() - start_ts
                            metrics = {"latency_sec": f"{duration:.2f}", "provider": "Cloud (Fallback)"}
                            self.notebook.log_exchange(user_input, full_response, current_model, metrics)
                        except Exception as cloud_e:
                            print(f"\n[Fallback Failed]: {cloud_e}")
                            self.notebook.log_error(cloud_e, "Cloud fallback")

        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
        except SystemExit:
            pass
        finally:
            self.graceful_shutdown()

    def handle_switch(self, current_provider):
        """Switch between local and cloud providers"""
        if current_provider == self.local:
            new_provider = self.cloud
            new_model = self.config.cloud_model
            print(f"\n🔄 Switching to Cloud ({new_model})")
        else:
            new_provider = self.local
            new_model = self.config.default_local_model
            print(f"\n🔄 Switching to Local ({new_model})")

        self.model_switches += 1
        self.notebook.log_system_note("Manual Switch", f"Switched from {current_provider.name} to {new_provider.name}")
        self.explain(f"Context switched. History is preserved in provider memory.")
        return new_provider, new_model

    def show_models(self):
        """Display available models"""
        models = self.local.list_models()
        if models:
            print("\n📚 Available Local Models:")
            for m in models:
                print(f"  • {m['name']} ({m['family']}, {m['size']})")
        else:
            print("No local models found or Ollama not running")

    def run_doctor(self):
        """Diagnose system health and GPU usage"""
        print("\n🩺 Maestro System Doctor")
        print("-----------------------")

        # 1. Check Role
        print("1. Architecture Check:")
        print("   • Maestro is a CLIENT application.")
        print("   • It does not perform inference itself.")
        print("   • GPU usage should be observed in the BACKEND (Ollama/Docker).")

        # 2. Local Backend Check
        print("\n2. Local Backend (Ollama) Check:")
        try:
            resp = requests.get(f"{self.config.ollama_url}/api/version", timeout=2)
            if resp.status_code == 200:
                print(f"   • Service Status: ONLINE (v{resp.json().get('version', 'unknown')})")

                # Tip for GPU
                print("\n   💡 GPU Troubleshooting Tip:")
                print("   If your GPU is idle during generation, check:")
                print("   1. Is 'ollama serve' running?")
                print("   2. Did Ollama detect your GPU at startup? (Check server logs)")
                print("   3. Is the model too large for VRAM? (Ollama may fall back to CPU)")
                print("   4. Try setting environment variable: OLLAMA_NUM_GPU=999")
            else:
                print(f"   • Service Status: UNREACHABLE ({resp.status_code})")
        except Exception as e:
            print(f"   • Service Status: ERROR ({e})")
            print("   • Verify Ollama is running at", self.config.ollama_url)

    def graceful_shutdown(self):
        """Clean session closure with summary"""
        duration = datetime.now() - self.start_time
        summary = {
            'duration': str(duration).split('.')[0],
            'switches': self.model_switches
        }

        if self.notebook:
            self.notebook.write_footer(summary)

        self.logger.info(f"Session ended. Duration: {summary['duration']}")
        self.logger.info(f"Notebook saved: {self.notebook.notebook_path}")


# ==================== CLI Entry Point ====================

def main():
    parser = argparse.ArgumentParser(description='Maestro Hybrid AI Orchestrator')

    parser.add_argument('--mode', choices=['local', 'cloud', 'hybrid'],
                       default='hybrid', help='Inference mode')
    parser.add_argument('--session', type=str, help='Session name for notebook')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    parser.add_argument('--quiet', action='store_true', help='Minimal console output')
    parser.add_argument('--explain', action='store_true', help='Show internal logic and decisions')

    args = parser.parse_args()

    # Setup
    config = Config.from_file(args.config) if args.config else Config()
    logger = setup_logging(args.verbose and not args.quiet)

    # Run
    maestro = Maestro(config, logger, mode=args.mode, explain_mode=args.explain)
    maestro.initialize_session(args.session)
    maestro.chat_loop()


if __name__ == "__main__":
    main()
