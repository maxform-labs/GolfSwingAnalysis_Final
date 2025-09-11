"""
골프 스윙 분석 시스템 v4.2 - 메인 패키지
820fps 고속 촬영 기반 수직 스테레오 비전 골프 스윙 분석 시스템
"""

__version__ = "4.2.0"
__author__ = "Maxform Labs"
__description__ = "Golf Swing Analysis System with 820fps Stereo Vision"

# 주요 모듈 import
from . import core
from . import algorithms
from . import analyzers
from . import processing
from . import interfaces
from . import validation
from . import utils

__all__ = [
    'core',
    'algorithms', 
    'analyzers',
    'processing',
    'interfaces',
    'validation',
    'utils'
]