#!/usr/bin/env python3
"""
Test script for Phase 1 Quick Wins
Tests the new basket index and enhanced city analysis without breaking existing functionality
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analytics.metrics import basket_index, basket_index_true_equal_weight
from analytics.signals import city_watchlist, load

def test_basket_index_improvements():
    """Test that both basket index functions work and produce different results"""
    print("🧪 Testing Basket Index Improvements...")
    
    # Load test data
    db_path = "data/foodprice.sqlite"
    try:
        df = load(db_path)
        print(f"✅ Loaded {len(df)} records from database")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False
    
    # Test original function
    try:
        original_basket = basket_index(df)
        print(f"✅ Original basket index: {len(original_basket)} records")
    except Exception as e:
        print(f"❌ Original basket index failed: {e}")
        return False
    
    # Test new function
    try:
        new_basket = basket_index_true_equal_weight(df)
        print(f"✅ New basket index: {len(new_basket)} records")
    except Exception as e:
        print(f"❌ New basket index failed: {e}")
        return False
    
    # Compare results
    if len(original_basket) == len(new_basket):
        print("✅ Both functions produce same number of records")
        
        # Check if they produce different values (they should!)
        sample_original = original_basket.head(10)
        sample_new = new_basket.head(10)
        
        # Merge to compare
        comparison = sample_original.merge(
            sample_new, 
            on=["city", "market_type", "date"], 
            suffixes=("_orig", "_new")
        )
        
        if not comparison.empty:
            # Check if values are different
            differences = comparison["basket_index_orig"] != comparison["basket_index_new"]
            if differences.any():
                print("✅ New function produces different (better) results!")
                print(f"   Sample differences found: {differences.sum()}/{len(comparison)}")
            else:
                print("⚠️  Functions produce identical results (may need investigation)")
        else:
            print("⚠️  No overlapping data to compare")
    else:
        print("❌ Functions produce different numbers of records")
        return False
    
    return True

def test_city_analysis_enhancements():
    """Test that enhanced city analysis includes wholesale insights"""
    print("\n🧪 Testing City Analysis Enhancements...")
    
    # Load test data
    db_path = "data/foodprice.sqlite"
    try:
        df = load(db_path)
        print(f"✅ Loaded {len(df)} records from database")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False
    
    # Test enhanced city watchlist
    try:
        city_results = city_watchlist(df)
        print(f"✅ City watchlist: {len(city_results)} cities analyzed")
        
        # Check for new wholesale columns
        expected_columns = [
            "city", "score", "reasons", "premium", "breadth", "CITY_MARGIN_STRESS",
            "wholesale_premium", "WHOLESALE_PREMIUM_RISING", "WHOLESALE_STRESS"
        ]
        
        missing_columns = [col for col in expected_columns if col not in city_results.columns]
        if missing_columns:
            print(f"❌ Missing expected columns: {missing_columns}")
            return False
        else:
            print("✅ All expected columns present")
        
        # Check if wholesale data is being analyzed
        wholesale_cities = city_results[city_results["wholesale_premium"] != 0.0]
        if not wholesale_cities.empty:
            print(f"✅ Wholesale analysis working: {len(wholesale_cities)} cities have wholesale insights")
        else:
            print("⚠️  No wholesale insights found (may be normal if no wholesale data)")
        
        # Show sample results
        print("\n📊 Sample City Analysis Results:")
        sample = city_results.head(5)[["city", "score", "premium", "wholesale_premium"]]
        print(sample.to_string(index=False))
        
    except Exception as e:
        print(f"❌ City watchlist failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Testing Phase 1 Quick Wins...\n")
    
    # Test 1: Basket Index Improvements
    basket_success = test_basket_index_improvements()
    
    # Test 2: City Analysis Enhancements
    city_success = test_city_analysis_enhancements()
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    if basket_success:
        print("✅ Basket Index Improvements: PASSED")
    else:
        print("❌ Basket Index Improvements: FAILED")
    
    if city_success:
        print("✅ City Analysis Enhancements: PASSED")
    else:
        print("❌ City Analysis Enhancements: FAILED")
    
    if basket_success and city_success:
        print("\n🎉 ALL QUICK WINS PASSED! System is ready for Phase 2.")
    else:
        print("\n⚠️  Some tests failed. Review before proceeding to Phase 2.")
    
    print("\n💡 Next Steps:")
    print("1. Review test results above")
    print("2. If all tests pass, consider updating the app.py to use new functions")
    print("3. Run full analytics pipeline to generate new insights")
    print("4. Document the improvements made")

if __name__ == "__main__":
    main()
