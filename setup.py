from setuptools import setup, find_packages

setup(
    name="telco-churn-experimentation",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.2.0",
        "scipy>=1.7.0",
        "Jinja2>=3.0.0",
        "streamlit>=1.20.0",
    ],
)
