#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test of Enhanced Adaptive Golf Measurement System
Tests single shot to validate multi-stage ROI performance
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import glob

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_adaptive_system import EnhancedAdaptiveSystem

def test_single_shot():
    """Test adaptive ROI system with single shot"""
    
    print("=== Quick Adaptive ROI Test ===")
    print("Testing single shot to validate multi-stage detection")
    print("="*50)
    
    # Initialize system
    system = EnhancedAdaptiveSystem()
    
    # Test with logo_ball-1 (English path)
    shot_folder = "C:/src/GolfSwingAnalysis_Final_ver8/shot-image/7iron/logo_ball-1"
    
    if not os.path.exists(shot_folder):
        print(f"Shot folder not found: {shot_folder}")
        return None
    
    print(f"Processing: {shot_folder}")
    
    # Process single shot
    from pathlib import Path
    results = system.analyze_shot_sequence(Path(shot_folder))
    
    if not results:
        print("No results generated")
        return None
    
    # Analyze results
    print(f"\n=== RESULTS ANALYSIS ===")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame([r.__dict__ for r in results])
    
    detected_frames = df[df['ball_detected'] == 1]
    total_frames = len(df)
    detection_rate = len(detected_frames) / total_frames * 100
    
    print(f"Detection: {len(detected_frames)}/{total_frames} frames ({detection_rate:.1f}%)")
    
    if len(detected_frames) > 0:
        # Method analysis
        method_counts = detected_frames['detection_method'].value_counts()
        print("Methods used:")
        for method, count in method_counts.items():
            print(f"  {method}: {count} frames")
        
        # Phase analysis
        phase_counts = detected_frames['detection_phase'].value_counts()
        print("Phases used:")
        for phase, count in phase_counts.items():
            print(f"  {phase}: {count} frames")
        
        # Confidence analysis
        avg_confidence = detected_frames['detection_confidence'].mean()
        print(f"Average detection confidence: {avg_confidence:.3f}")
        
        if 'confidence_score' in detected_frames.columns:
            overall_confidence = detected_frames['confidence_score'].mean()
            print(f"Average overall confidence: {overall_confidence:.3f}")
        
        # Physics analysis
        valid_speeds = detected_frames[detected_frames['ball_speed_mph'] > 0]
        if len(valid_speeds) > 0:
            print(f"Ball speed: {valid_speeds['ball_speed_mph'].mean():.1f} mph")
        
        valid_angles = detected_frames[detected_frames['launch_angle_deg'] != 0]
        if len(valid_angles) > 0:
            print(f"Launch angle: {valid_angles['launch_angle_deg'].mean():.1f}°")
    
    # Save quick results
    output_file = "results/quick_adaptive_test_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".xlsx"
    os.makedirs("results", exist_ok=True)
    
    df.to_excel(output_file, index=False)
    print(f"\nResults saved: {output_file}")
    
    return df

if __name__ == "__main__":
    results = test_single_shot()
    
    if results is not None:
        print("\n=== TEST COMPLETED SUCCESSFULLY ===")
        
        # Show key metrics
        detected = results[results['ball_detected'] == 1]
        if len(detected) > 0:
            print(f"SUCCESS: Multi-stage ROI detected {len(detected)} frames")
            print(f"Best method: {detected['detection_method'].mode().iloc[0]}")
            print(f"Most effective phase: {detected['detection_phase'].mode().iloc[0]}")
        else:
            print("WARNING: No frames detected - check image brightness/paths")
    else:
        print("TEST FAILED: No results generated")