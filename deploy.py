#!/usr/bin/env python3
"""
Deployment Script for Stroke Severity Prediction Model
Provides easy access to different deployment options.
"""

import argparse
import subprocess
import sys
import os

def run_api():
    """Run the Flask API server."""
    print("🚀 Starting Flask API Server...")
    print("📡 API will be available at: http://localhost:5000")
    print("📖 API Documentation: http://localhost:5000/example")
    print("🔍 Health Check: http://localhost:5000/health")
    print("\nPress Ctrl+C to stop the server\n")

    try:
        subprocess.run([sys.executable, "api.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start API server: {e}")

def run_dashboard():
    """Run the Streamlit dashboard."""
    print("🚀 Starting Streamlit Dashboard...")
    print("🌐 Dashboard will open in your browser")
    print("📱 Access at: http://localhost:8501")
    print("\nPress Ctrl+C to stop the dashboard\n")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start dashboard: {e}")

def run_cli_example():
    """Run CLI with example prediction."""
    print("🧠 Running CLI Example Prediction...")
    try:
        subprocess.run([
            sys.executable, "predict_cli.py",
            "--age", "65",
            "--nihss", "12",
            "--hypertension", "1"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ CLI example failed: {e}")

def run_service_test():
    """Test the prediction service."""
    print("🧪 Testing Prediction Service...")
    try:
        subprocess.run([sys.executable, "stroke_predictor_service.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Service test failed: {e}")

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def show_usage():
    """Show deployment options."""
    print("🧠 Stroke Severity Prediction Model - Deployment Options")
    print("=" * 60)
    print()
    print("Available deployment methods:")
    print()
    print("1. 🌐 Web Dashboard (Streamlit)")
    print("   Interactive web interface for predictions")
    print("   Command: python deploy.py --dashboard")
    print()
    print("2. 🔌 REST API (Flask)")
    print("   REST API for programmatic access")
    print("   Command: python deploy.py --api")
    print()
    print("3. 💻 Command Line Interface")
    print("   CLI tool for batch and single predictions")
    print("   Command: python predict_cli.py --help")
    print()
    print("4. 🧪 Service Test")
    print("   Test the prediction service directly")
    print("   Command: python deploy.py --test")
    print()
    print("5. 📦 Install Dependencies")
    print("   Install all required packages")
    print("   Command: python deploy.py --install")
    print()
    print("Examples:")
    print("  # Install dependencies")
    print("  python deploy.py --install")
    print()
    print("  # Run web dashboard")
    print("  python deploy.py --dashboard")
    print()
    print("  # Run API server")
    print("  python deploy.py --api")
    print()
    print("  # CLI single prediction")
    print("  python predict_cli.py --age 65 --nihss 12 --hypertension 1")
    print()
    print("  # CLI batch prediction")
    print("  python predict_cli.py --csv patients.csv --output results.csv")
    print()
    print("Files created:")
    print("  • stroke_predictor_service.py - Core prediction service")
    print("  • api.py - Flask REST API")
    print("  • dashboard.py - Streamlit web dashboard")
    print("  • predict_cli.py - Command-line interface")
    print()

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(
        description="Stroke Severity Prediction Model Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--api', action='store_true',
                       help='Run Flask REST API server')
    parser.add_argument('--dashboard', action='store_true',
                       help='Run Streamlit web dashboard')
    parser.add_argument('--cli-example', action='store_true',
                       help='Run CLI with example prediction')
    parser.add_argument('--test', action='store_true',
                       help='Test the prediction service')
    parser.add_argument('--install', action='store_true',
                       help='Install required dependencies')

    args = parser.parse_args()

    # If no arguments provided, show usage
    if not any([args.api, args.dashboard, args.cli_example, args.test, args.install]):
        show_usage()
        return

    # Handle different deployment options
    if args.install:
        install_dependencies()
    elif args.api:
        run_api()
    elif args.dashboard:
        run_dashboard()
    elif args.cli_example:
        run_cli_example()
    elif args.test:
        run_service_test()

if __name__ == "__main__":
    main()