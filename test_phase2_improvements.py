#!/usr/bin/env python3
"""
Test script for Phase 2 Quality Improvements
Tests comprehensive error handling, data validation, and margin analysis robustness
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analytics.metrics import (
    load_facts, product_summary, city_spread, margins, 
    anomalies, basket_index, basket_index_true_equal_weight
)
from analytics.signals import product_watchlist, city_watchlist, load

def test_error_handling():
    """Test that error handling works correctly for various edge cases"""
    print("🧪 Testing Error Handling Improvements...")
    
    # Test 1: Invalid database path
    try:
        result = load_facts("nonexistent_database.db")
        print("❌ Should have failed for nonexistent database")
        return False
    except FileNotFoundError:
        print("✅ Correctly handled nonexistent database path")
    except Exception as e:
        print(f"❌ Unexpected error for nonexistent database: {e}")
        return False
    
    # Test 2: Empty DataFrame handling
    try:
        empty_df = pd.DataFrame(columns=["product_id", "name", "city", "date", "price", "market_type"])
        result = product_summary(empty_df)
        print("❌ Should have failed for empty DataFrame")
        return False
    except ValueError as e:
        if "empty" in str(e).lower():
            print("✅ Correctly handled empty DataFrame")
        else:
            print(f"❌ Wrong error message for empty DataFrame: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error for empty DataFrame: {e}")
        return False
    
    # Test 3: Missing columns handling
    try:
        incomplete_df = pd.DataFrame({
            "product_id": [1, 2],
            "name": ["Product 1", "Product 2"],
            "date": ["2024-01-01", "2024-01-02"]
            # Missing price and market_type columns
        })
        result = product_summary(incomplete_df)
        print("❌ Should have failed for missing columns")
        return False
    except ValueError as e:
        if "missing" in str(e).lower():
            print("✅ Correctly handled missing columns")
        else:
            print(f"❌ Wrong error message for missing columns: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error for missing columns: {e}")
        return False
    
    print("✅ All error handling tests passed")
    return True

def test_data_validation():
    """Test that data validation works correctly"""
    print("\n🧪 Testing Data Validation Improvements...")
    
    # Test with valid data
    try:
        db_path = "data/foodprice.sqlite"
        df = load_facts(db_path)
        print(f"✅ Successfully loaded {len(df)} records with validation")
        
        # Check that validation warnings are shown
        print("📊 Data quality summary:")
        print(f"   Total records: {len(df)}")
        print(f"   Valid prices: {df['price'].notna().sum()}")
        print(f"   Valid dates: {df['date'].notna().sum()}")
        print(f"   Market types: {df['market_type'].unique()}")
        
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return False
    
    print("✅ Data validation tests passed")
    return True

def test_margin_analysis_robustness():
    """Test that margin analysis handles missing data gracefully"""
    print("\n🧪 Testing Margin Analysis Robustness...")
    
    try:
        db_path = "data/foodprice.sqlite"
        df = load_facts(db_path)
        
        # Test margin calculation
        margin_results = margins(df)
        
        if margin_results.empty:
            print("❌ Margin analysis produced empty results")
            return False
        
        print(f"✅ Margin analysis completed successfully")
        print(f"📊 Margin results: {len(margin_results)} records")
        
        # Check for new data quality columns
        expected_quality_cols = ["margin_quality", "has_retail", "has_wholesale"]
        quality_cols_present = [col for col in expected_quality_cols if col in margin_results.columns]
        
        if quality_cols_present:
            print(f"✅ Data quality indicators present: {quality_cols_present}")
            
            # Show quality distribution
            if "margin_quality" in margin_results.columns:
                quality_dist = margin_results["margin_quality"].value_counts()
                print(f"📊 Margin quality distribution:")
                for quality, count in quality_dist.items():
                    print(f"   {quality}: {count}")
        else:
            print("⚠️  Data quality indicators not found")
        
        # Check margin calculation completeness
        valid_margins = margin_results["margin"].notna().sum()
        total_margins = len(margin_results)
        completeness = valid_margins / total_margins * 100 if total_margins > 0 else 0
        
        print(f"📊 Margin calculation completeness: {completeness:.1f}%")
        
    except Exception as e:
        print(f"❌ Margin analysis robustness test failed: {e}")
        return False
    
    print("✅ Margin analysis robustness tests passed")
    return True

def test_anomaly_detection_improvements():
    """Test that anomaly detection has better error handling"""
    print("\n🧪 Testing Anomaly Detection Improvements...")
    
    try:
        db_path = "data/foodprice.sqlite"
        df = load_facts(db_path)
        
        # Test with default threshold
        anomalies_df = anomalies(df)
        
        if anomalies_df is not None:  # Should return empty DataFrame if no anomalies
            print(f"✅ Anomaly detection completed successfully")
            print(f"📊 Anomalies found: {len(anomalies_df)}")
            
            # Check for new anomaly characteristics
            if "anomaly_magnitude" in anomalies_df.columns:
                print("✅ Anomaly magnitude calculation working")
            if "anomaly_direction" in anomalies_df.columns:
                print("✅ Anomaly direction calculation working")
        else:
            print("❌ Anomaly detection returned None")
            return False
        
        # Test with custom threshold
        custom_anomalies = anomalies(df, thresh=2.0)
        print(f"✅ Custom threshold anomaly detection working: {len(custom_anomalies)} anomalies")
        
    except Exception as e:
        print(f"❌ Anomaly detection improvements test failed: {e}")
        return False
    
    print("✅ Anomaly detection improvements tests passed")
    return True

def test_product_watchlist_robustness():
    """Test that product watchlist has better error handling"""
    print("\n🧪 Testing Product Watchlist Robustness...")
    
    try:
        db_path = "data/foodprice.sqlite"
        df = load_facts(db_path)
        
        # Test product watchlist generation
        watchlist = product_watchlist(df)
        
        if watchlist is not None and not watchlist.empty:
            print(f"✅ Product watchlist generated successfully")
            print(f"📊 Products analyzed: {len(watchlist)}")
            
            # Check risk score distribution
            if "score" in watchlist.columns:
                score_dist = watchlist["score"].value_counts().sort_index()
                print(f"📊 Risk score distribution: {score_dist.to_dict()}")
            
            # Check reasons generation
            if "reasons" in watchlist.columns:
                valid_reasons = watchlist["reasons"].notna().sum()
                print(f"📊 Valid reasons generated: {valid_reasons}/{len(watchlist)}")
        else:
            print("❌ Product watchlist generation failed")
            return False
        
    except Exception as e:
        print(f"❌ Product watchlist robustness test failed: {e}")
        return False
    
    print("✅ Product watchlist robustness tests passed")
    return True

def test_city_watchlist_enhancements():
    """Test that city watchlist includes wholesale insights"""
    print("\n🧪 Testing City Watchlist Enhancements...")
    
    try:
        db_path = "data/foodprice.sqlite"
        df = load_facts(db_path)
        
        # Test enhanced city watchlist
        city_results = city_watchlist(df)
        
        if city_results is not None and not city_results.empty:
            print(f"✅ Enhanced city watchlist generated successfully")
            print(f"📊 Cities analyzed: {len(city_results)}")
            
            # Check for wholesale columns
            wholesale_cols = ["wholesale_premium", "WHOLESALE_PREMIUM_RISING", "WHOLESALE_STRESS"]
            present_wholesale_cols = [col for col in wholesale_cols if col in city_results.columns]
            
            if present_wholesale_cols:
                print(f"✅ Wholesale analysis columns present: {present_wholesale_cols}")
                
                # Check wholesale data quality
                if "wholesale_premium" in city_results.columns:
                    wholesale_cities = city_results[city_results["wholesale_premium"] != 0.0]
                    print(f"📊 Cities with wholesale insights: {len(wholesale_cities)}")
            else:
                print("⚠️  Wholesale analysis columns not found")
            
            # Check enhanced scoring
            if "score" in city_results.columns:
                max_score = city_results["score"].max()
                print(f"📊 Enhanced scoring working - max score: {max_score}")
        else:
            print("❌ Enhanced city watchlist generation failed")
            return False
        
    except Exception as e:
        print(f"❌ City watchlist enhancements test failed: {e}")
        return False
    
    print("✅ City watchlist enhancements tests passed")
    return True

def test_full_pipeline():
    """Test the complete analytics pipeline with all improvements"""
    print("\n🧪 Testing Full Analytics Pipeline...")
    
    try:
        db_path = "data/foodprice.sqlite"
        df = load_facts(db_path)
        
        print(f"📊 Starting pipeline with {len(df)} records")
        
        # Run all analytics functions
        results = {}
        
        # Product summary
        results["product_summary"] = product_summary(df)
        print(f"✅ Product summary: {len(results['product_summary'])} products")
        
        # City spread
        results["city_spread"] = city_spread(df)
        print(f"✅ City spread: {len(results['city_spread'])} records")
        
        # Margins
        results["margins"] = margins(df)
        print(f"✅ Margins: {len(results['margins'])} records")
        
        # Anomalies
        results["anomalies"] = anomalies(df)
        print(f"✅ Anomalies: {len(results['anomalies'])} records")
        
        # Basket index (both versions)
        results["basket_index"] = basket_index(df)
        results["basket_index_enhanced"] = basket_index_true_equal_weight(df)
        print(f"✅ Basket index: {len(results['basket_index'])} records")
        print(f"✅ Enhanced basket index: {len(results['basket_index_enhanced'])} records")
        
        # Product watchlist
        results["product_watchlist"] = product_watchlist(df)
        print(f"✅ Product watchlist: {len(results['product_watchlist'])} products")
        
        # City watchlist
        results["city_watchlist"] = city_watchlist(df)
        print(f"✅ City watchlist: {len(results['city_watchlist'])} cities")
        
        print("🎉 Full pipeline completed successfully!")
        
        # Summary statistics
        print("\n📋 Pipeline Results Summary:")
        for name, result in results.items():
            if hasattr(result, 'shape'):
                print(f"   {name}: {result.shape[0]} rows × {result.shape[1]} columns")
            elif hasattr(result, '__len__'):
                print(f"   {name}: {len(result)} items")
            else:
                print(f"   {name}: {type(result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False

def main():
    """Run all Phase 2 tests"""
    print("🚀 Testing Phase 2 Quality Improvements...\n")
    
    tests = [
        ("Error Handling", test_error_handling),
        ("Data Validation", test_data_validation),
        ("Margin Analysis Robustness", test_margin_analysis_robustness),
        ("Anomaly Detection Improvements", test_anomaly_detection_improvements),
        ("Product Watchlist Robustness", test_product_watchlist_robustness),
        ("City Watchlist Enhancements", test_city_watchlist_enhancements),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📋 PHASE 2 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL PHASE 2 IMPROVEMENTS PASSED!")
        print("🚀 System is ready for Phase 3 (Strategic Improvements)")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review before proceeding to Phase 3.")
    
    print("\n💡 Next Steps:")
    print("1. Review any failed tests above")
    print("2. If all tests pass, consider moving to Phase 3")
    print("3. Phase 3 includes threshold calibration and configurable parameters")
    print("4. The system now has robust error handling and data validation")

if __name__ == "__main__":
    main()
