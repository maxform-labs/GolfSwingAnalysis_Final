#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Golf Swing Analysis System v4.2 Setup Configuration
"""

from setuptools import setup, find_packages
import os

# Read the long description from README
long_description = "Golf Swing Analysis System v4.2 - 820fps high-speed stereo vision based real-time golf swing analysis"

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    try:
        with open(requirements_path, 'r', encoding='utf-8') as f:
            requirements = []
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Handle inline comments
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    requirements.append(line)
            return requirements
    except FileNotFoundError:
        return []

setup(
    name="golf-swing-analysis",
    version="4.2.0",
    author="Maxform Development Team",
    author_email="dev@maxform.com",
    description="820fps high-speed stereo vision based golf swing analysis system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxform/golf-swing-analysis",
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    extras_require={
        'cuda': ['cupy-cuda11x>=12.2.0'],
        'dev': ['pytest>=7.4.0', 'black>=23.0.0', 'flake8>=6.0.0'],
        'web': ['playwright>=1.37.0'],
    },
    python_requires='>=3.8',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Sports/Golf Industry",
        "Topic :: Scientific/Engineering :: Computer Vision",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
    ],
    entry_points={
        'console_scripts': [
            'golf-analyzer=scripts.run_main_analyzer:main',
            'golf-kiosk=scripts.run_kiosk_system:main',
            'golf-web=scripts.run_web_dashboard:main',
            'golf-validate=scripts.run_accuracy_validator:main',
        ],
    },
    keywords="golf swing analysis computer-vision stereo-vision 820fps sports-analytics",
    project_urls={
        "Bug Reports": "https://github.com/maxform/golf-swing-analysis/issues",
        "Source": "https://github.com/maxform/golf-swing-analysis",
        "Documentation": "https://github.com/maxform/golf-swing-analysis/docs",
    },
)