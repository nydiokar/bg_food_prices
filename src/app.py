import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from analytics.signals import load, product_watchlist, city_watchlist
from analytics.metrics import city_spread, margins, anomalies, basket_index
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
DB   = ROOT / "data" / "foodprice.sqlite"

# Page config
st.set_page_config(
    layout="wide", 
    page_title="BG Price Intelligence",
    page_icon="📊"
)

# Load data
@st.cache_data
def load_data():
    df = load(str(DB))
    prod_sig = product_watchlist(df)
    city_sig = city_watchlist(df)
    return df, prod_sig, city_sig

df, prod_sig, city_sig = load_data()

# Helper functions for advanced analytics
def get_city_correlations(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Find cities that move together in price patterns"""
    # Get city basket indices
    bi = basket_index(df)
    city_baskets = bi[bi["market_type"]=="retail"].pivot(index="date", columns="city", values="basket_index")
    
    # Calculate correlations
    corr_matrix = city_baskets.corr()
    
    # Get top correlations (excluding self-correlation)
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            city1 = corr_matrix.columns[i]
            city2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if not pd.isna(corr_val):
                correlations.append({
                    'city1': city1,
                    'city2': city2,
                    'correlation': corr_val
                })
    
    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    return corr_df.head(top_n)

def get_product_correlations(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Find products that move together in price"""
    # Get product price changes
    nat = df.groupby(["product_id", "name", "date"])["price"].median().reset_index()
    nat["ret"] = nat.groupby("product_id")["price"].pct_change(fill_method=None)
    
    # Pivot to get returns matrix
    ret_matrix = nat.pivot(index="date", columns="product_id", values="ret")
    
    # Calculate correlations
    corr_matrix = ret_matrix.corr()
    
    # Get top correlations
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            prod1_id = corr_matrix.columns[i]
            prod2_id = corr_matrix.columns[j]
            prod1_name = nat[nat["product_id"]==prod1_id]["name"].iloc[0]
            prod2_name = nat[nat["product_id"]==prod2_id]["name"].iloc[0]
            corr_val = corr_matrix.iloc[i, j]
            if not pd.isna(corr_val):
                correlations.append({
                    'product1': prod1_name,
                    'product2': prod2_name,
                    'correlation': corr_val
                })
    
    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    return corr_df.head(top_n)

# Clean, minimal header
st.title("📊 Bulgarian Food Price Intelligence")
st.markdown("*Market signals, trends, and opportunities across Bulgaria*")

# Market Overview at the top for context
st.header("📈 Market Overview")
bi = basket_index(df)
national_retail = bi[bi["market_type"]=="retail"].groupby("date")["basket_index"].median().reset_index()

# Create a compact market overview
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # Compact chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=national_retail["date"], y=national_retail["basket_index"],
        mode='lines',
        name='National Basket',
        line=dict(color='#ff7f0e', width=2)
    ))
    fig.add_hline(y=100, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="National Food Price Trend (Base=100)",
        height=300,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=10),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False)
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    if len(national_retail) > 1:
        latest_national = national_retail["basket_index"].iloc[-1]
        national_change = ((latest_national - 100) / 100) * 100
        
        st.metric("Current Index", f"{latest_national:.1f}")
        st.metric("vs Baseline", f"{national_change:+.1f}%")

with col3:
    if len(national_retail) > 1:
        trend_direction = "📈 Rising" if national_change > 0 else "📉 Falling"
        st.metric("Trend", trend_direction)
        
        # Quick stats
        st.markdown("**Quick Stats:**")
        st.markdown(f"• Data points: {len(national_retail)}")
        st.markdown(f"• Volatility: {national_retail['basket_index'].std():.1f}")

st.divider()

# Toggle between Hot Products and Cities
st.header("🚨 Market Signals")
watchlist_type = st.radio("Select watchlist:", ["🔥 Hot Products", "🏙️ City Alerts"], horizontal=True)

if watchlist_type == "🔥 Hot Products":
    st.subheader("Products with unusual price activity")
    
    # Enhanced product display with better formatting
    display_prod = prod_sig.head(20).copy()
    display_prod["yoy_pct"] = (display_prod["yoy"] * 100).round(1)
    display_prod["trend"] = display_prod["slope_8w"].apply(
        lambda x: "📈" if x > 0.001 else "📉" if x < -0.001 else "➡️"
    )
    
    # Color-code by score
    def color_score(val):
        if val >= 5: return "🔴"
        elif val >= 3: return "🟡" 
        else: return "🟢"
    
    display_prod["alert"] = display_prod["score"].apply(color_score)
    
    # Convert technical risk factors to human-readable
    def explain_risk_factors(reasons_str):
        if pd.isna(reasons_str) or reasons_str == "":
            return "No specific risks detected"
        
        explanations = []
        if "HEAT_YOY" in reasons_str:
            explanations.append("🔥 High YoY inflation")
        if "HEAT_MOM_PERSIST" in reasons_str:
            explanations.append("📈 Persistent monthly growth")
        if "VOL_REGIME" in reasons_str:
            explanations.append("📊 Unusual volatility")
        if "ELEVATED_LEVEL" in reasons_str:
            explanations.append("⬆️ Above normal price levels")
        if "REVERSAL_UP" in reasons_str:
            explanations.append("🔄 Trend reversal upward")
        if "MARGIN_STRESS" in reasons_str:
            explanations.append("💸 Margin pressure")
        
        return " | ".join(explanations) if explanations else "Technical indicators"
    
    display_prod["risk_explanation"] = display_prod["reasons"].apply(explain_risk_factors)
    
    # Compact table with better column widths
    st.dataframe(
        display_prod[["alert", "name", "score", "trend", "yoy_pct", "risk_explanation"]].assign(
            yoy_pct=lambda d: d["yoy_pct"].astype(str) + "%"
        ),
        hide_index=True,
        column_config={
            "alert": st.column_config.TextColumn("Alert", width="small"),
            "name": st.column_config.TextColumn("Product", width="medium"),
            "score": st.column_config.NumberColumn("Risk Score", width="small"),
            "trend": st.column_config.TextColumn("Trend", width="small"),
            "yoy_pct": st.column_config.TextColumn("YoY %", width="small"),
            "risk_explanation": st.column_config.TextColumn("Risk Explanation", width="large")
        }
    )

else:
    st.subheader("Cities with unusual price patterns")
    
    display_city = city_sig.head(20).copy()
    display_city["premium_pct"] = (display_city["premium"] * 100).round(1)
    display_city["breadth_pct"] = (display_city["breadth"] * 100).round(0)
    
    # Add stress indicators
    def stress_indicator(row):
        if row["CITY_MARGIN_STRESS"]: return "🔴"
        elif row["premium"] > 0.05: return "🟡"
        else: return "🟢"
    
    display_city["stress"] = display_city.apply(stress_indicator, axis=1)
    
    # Generate meaningful city risk explanations
    def explain_city_risks(row):
        risks = []
        if row["CITY_MARGIN_STRESS"]:
            risks.append("💸 High margin stress")
        if row["premium"] > 0.05:
            risks.append(f"💰 {row['premium_pct']:.1f}% above national average")
        if row["breadth"] > 0.3:
            risks.append(f"📊 {row['breadth_pct']:.0f}% of products stressed")
        
        if not risks:
            return "✅ Stable price environment"
        return " | ".join(risks)
    
    display_city["risk_explanation"] = display_city.apply(explain_city_risks, axis=1)
    
    st.dataframe(
        display_city[["stress", "city", "premium_pct", "breadth_pct", "risk_explanation"]].assign(
            premium_pct=lambda d: d["premium_pct"].astype(str) + "%",
            breadth_pct=lambda d: d["breadth_pct"].astype(str) + "%"
        ),
        hide_index=True,
        column_config={
            "stress": st.column_config.TextColumn("Alert", width="small"),
            "city": st.column_config.TextColumn("City", width="medium"),
            "premium_pct": st.column_config.TextColumn("Price Premium", width="small"),
            "breadth_pct": st.column_config.TextColumn("Unusual Products %", width="small"),
            "risk_explanation": st.column_config.TextColumn("Risk Explanation", width="large")
        }
    )

st.divider()

# Simplified Deep Dive Analysis
st.header("🔍 Deep Dive Analysis")

tab1, tab2, tab3 = st.tabs(["📈 Product Analysis", "🏙️ City Analysis", "🚨 Anomalies"])

with tab1:
    st.subheader("Product Price Trends & Margins")
    
    # Product selector with search
    pmap = {pid: n for pid, n in df[["product_id","name"]].dropna().drop_duplicates().values}
    search = st.text_input("🔍 Search products:", placeholder="Type product name...")
    
    if search:
        filtered_products = {k: v for k, v in pmap.items() if search.lower() in v.lower()}
    else:
        filtered_products = pmap
    
    if filtered_products:
        pid = st.selectbox("Select Product:", sorted(filtered_products.keys()), 
                          format_func=lambda k: filtered_products[k])
        
        if pid:
            product_name = pmap[pid]
            
            # Get product data for spread analysis (percentiles)
            sp = city_spread(df[df["product_id"]==pid]).sort_values("date")
            
            # Get both retail and wholesale data for comparison
            retail_data = df[(df["product_id"]==pid) & (df["market_type"]=="retail")].groupby("date")["price"].median().reset_index()
            wholesale_data = df[(df["product_id"]==pid) & (df["market_type"]=="wholesale")].groupby("date")["price"].median().reset_index()
            
            # Create comparison chart with percentiles
            fig = go.Figure()
            
            # Retail prices
            if not retail_data.empty:
                fig.add_trace(go.Scatter(
                    x=retail_data["date"], y=retail_data["price"],
                    mode='lines+markers',
                    name='Retail Price (National Median)',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=6)
                ))
            
            # Wholesale prices
            if not wholesale_data.empty:
                fig.add_trace(go.Scatter(
                    x=wholesale_data["date"], y=wholesale_data["price"],
                    mode='lines+markers',
                    name='Wholesale Price (National Median)',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=6)
                ))
            
            # Add percentile bands if we have spread data
            if not sp.empty:
                fig.add_trace(go.Scatter(
                    x=sp["date"], y=sp["p90"],
                    mode='lines',
                    name='90th Percentile (High) - All Cities',
                    line=dict(color='#ff6b6b', width=2, dash='dash'),
                    showlegend=True
                ))
                
                fig.add_trace(go.Scatter(
                    x=sp["date"], y=sp["p10"],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255, 107, 107, 0.2)',
                    name='10th Percentile (Low) - All Cities',
                    line=dict(color='#ff6b6b', width=2, dash='dash'),
                    showlegend=True
                ))
                
                # Add explanation
                st.info("💡 **Percentile Explanation:** 10th/90th percentiles show price spread across ALL cities. Wholesale prices are national medians and may be similar to 10th percentile because wholesale is often the lowest price tier.")
            
            fig.update_layout(
                title=f"Price Comparison: {product_name}",
                xaxis_title="Date",
                yaxis_title="Price (BGN)",
                height=400,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.3)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.3)')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Product insights
            if not sp.empty and len(sp) > 1:
                latest_price = sp["median"].iloc[-1]
                earliest_price = sp["median"].iloc[0]
                total_change = ((latest_price - earliest_price) / earliest_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"{latest_price:.2f} BGN")
                with col2:
                    st.metric("Total Change", f"{total_change:+.1f}%")
                with col3:
                    st.metric("Price Volatility", f"{sp['spread_rel'].mean():.1%}")
            
            # Margin trend chart
            if not retail_data.empty and not wholesale_data.empty:
                st.subheader("💰 Store Profit Margins")
                
                # Merge data and calculate margins
                merged_data = retail_data.merge(wholesale_data, on="date", suffixes=("_retail", "_wholesale"))
                merged_data["margin"] = merged_data["price_retail"] - merged_data["price_wholesale"]
                merged_data["margin_pct"] = (merged_data["margin"] / merged_data["price_wholesale"]) * 100
                
                # Create margin chart
                fig_margin = go.Figure()
                
                fig_margin.add_trace(go.Scatter(
                    x=merged_data["date"], y=merged_data["margin"],
                    mode='lines+markers',
                    name='Profit Margin (BGN)',
                    line=dict(color='#2ecc71', width=3),
                    marker=dict(size=6)
                ))
                
                fig_margin.add_hline(y=0, line_dash="dash", line_color="gray", 
                                   annotation_text="Break-even")
                
                fig_margin.update_layout(
                    title=f"Profit Margin Trend: {product_name}",
                    xaxis_title="Date",
                    yaxis_title="Margin (BGN)",
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                fig_margin.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.3)')
                fig_margin.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.3)')
                
                st.plotly_chart(fig_margin, use_container_width=True)
                
                # Margin insights - FIXED: Use consistent data source
                if len(merged_data) > 1:
                    latest_margin = merged_data["margin"].iloc[-1]
                    avg_margin = merged_data["margin"].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Margin", f"{latest_margin:.2f} BGN", 
                                 delta=f"{latest_margin - avg_margin:+.2f} BGN")
                    with col2:
                        st.metric("Average Margin", f"{avg_margin:.2f} BGN")
                    with col3:
                        st.metric("Margin Trend", "📈 Rising" if latest_margin > avg_margin else "📉 Falling")
                    
                    # Interpretation
                    if latest_margin > 0:
                        st.success(f"✅ Stores are currently making {latest_margin:.2f} BGN profit per unit")
                    else:
                        st.error(f"❌ Stores are currently losing {abs(latest_margin):.2f} BGN per unit")
                    
                    # Yearly margin breakdown - FIXED: Use same data source for consistency
                    st.subheader("📊 Yearly Margin Analysis")
                    
                    # Aggregate by year for cleaner visualization
                    merged_data["year"] = merged_data["date"].dt.year
                    yearly_margins = merged_data.groupby("year").agg({
                        "price_wholesale": "mean",
                        "price_retail": "mean",
                        "margin": "mean",
                        "margin_pct": "mean"
                    }).reset_index()
                    
                    if len(yearly_margins) > 1:
                        # Create yearly margin chart
                        fig_yearly = go.Figure()
                        
                        # Add margin bars
                        fig_yearly.add_trace(go.Bar(
                            x=yearly_margins["year"],
                            y=yearly_margins["margin"],
                            name='Profit Margin (BGN)',
                            marker_color=['#2ecc71' if x > 0 else '#e74c3c' for x in yearly_margins["margin"]],
                            text=[f"{x:.2f} BGN" for x in yearly_margins["margin"]],
                            textposition='auto'
                        ))
                        
                        fig_yearly.add_hline(y=0, line_dash="dash", line_color="gray", 
                                           annotation_text="Break-even (0 BGN)")
                        
                        fig_yearly.update_layout(
                            title=f"Yearly Profit Margins: {product_name}",
                            xaxis_title="Year",
                            yaxis_title="Margin (BGN)",
                            height=300,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        
                        fig_yearly.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.3)')
                        fig_yearly.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.3)')
                        
                        st.plotly_chart(fig_yearly, use_container_width=True)
                        
                        # Show yearly breakdown table
                        yearly_display = yearly_margins[["year", "price_wholesale", "price_retail", "margin", "margin_pct"]].copy()
                        yearly_display["price_wholesale"] = yearly_display["price_wholesale"].round(2).astype(str) + " BGN"
                        yearly_display["price_retail"] = yearly_display["price_retail"].round(2).astype(str) + " BGN"
                        yearly_display["margin"] = yearly_display["margin"].round(2).astype(str) + " BGN"
                        yearly_display["margin_pct"] = yearly_display["margin_pct"].round(1).astype(str) + "%"
                        
                        st.dataframe(
                            yearly_display,
                            hide_index=True,
                            column_config={
                                "year": "Year",
                                "price_wholesale": "Wholesale Price",
                                "price_retail": "Retail Price",
                                "margin": "Profit Margin",
                                "margin_pct": "Margin %"
                            }
                        )
                        
                        # Yearly margin interpretation - FIXED: Use consistent data
                        latest_year = yearly_margins["year"].iloc[-1]
                        latest_yearly_margin = yearly_margins["margin"].iloc[-1]
                        
                        # Add consistency check
                        if abs(latest_yearly_margin - latest_margin) < 0.01:  # Within 0.01 BGN
                            st.success(f"✅ In {latest_year}, stores made {latest_yearly_margin:.2f} BGN profit per unit (consistent with current margin)")
                        else:
                            st.warning(f"⚠️ In {latest_year}, stores made {latest_yearly_margin:.2f} BGN profit per unit (vs current: {latest_margin:.2f} BGN)")
                        
                        if latest_yearly_margin > 0:
                            st.success(f"✅ In {latest_year}, stores made {latest_yearly_margin:.2f} BGN profit per unit")
                        else:
                            st.error(f"❌ In {latest_year}, stores lost {abs(latest_yearly_margin):.2f} BGN per unit")
                    
                else:
                    st.info("Limited margin data available for this product")
                    
            else:
                st.info("No margin data available for this product (requires both wholesale and retail prices)")

with tab2:
    st.subheader("City Price Analysis")
    
    city = st.selectbox("Select City:", sorted(df["city"].unique()))
    
    if city:
        # Basket index analysis
        bi = basket_index(df)
        city_data = bi[(bi["city"]==city) & (bi["market_type"]=="retail")].sort_values("date")
        
        if not city_data.empty:
            # Create basket index chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=city_data["date"], y=city_data["basket_index"],
                mode='lines+markers',
                name=f'Basket Index - {city}',
                line=dict(color='#2ca02c', width=3),
                marker=dict(size=6)
            ))
            
            # Add baseline at 100
            fig.add_hline(y=100, line_dash="dash", line_color="gray", 
                         annotation_text="Baseline (100)")
            
            fig.update_layout(
                title=f"Price Basket Index: {city} (Base=100)",
                xaxis_title="Date",
                yaxis_title="Basket Index",
                height=400,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.3)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.3)')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # City insights
            latest_basket = city_data["basket_index"].iloc[-1]
            basket_change = ((latest_basket - 100) / 100) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Index", f"{latest_basket:.1f}")
            with col2:
                st.metric("vs Baseline", f"{basket_change:+.1f}%")
            with col3:
                st.metric("Data Points", len(city_data))
            
            # Show city's product mix
            st.subheader(f"Product Prices in {city}")
            city_products = df[df["city"]==city].groupby("name")["price"].last().sort_values(ascending=False)
            st.dataframe(
                city_products.reset_index().assign(
                    price=lambda d: d["price"].round(2).astype(str) + " BGN"
                ),
                hide_index=True,
                column_config={
                    "name": "Product",
                    "price": "Latest Price"
                }
            )

with tab3:
    st.subheader("🚨 Price Anomalies")
    
    # Get recent anomalies
    recent_anomalies = anomalies(df).sort_values("date", ascending=False).head(50)
    
    if not recent_anomalies.empty:
        # Add context to anomalies
        recent_anomalies["ret_pct"] = (recent_anomalies["ret"] * 100).round(2)
        recent_anomalies["price_formatted"] = recent_anomalies["price"].round(2).astype(str) + " BGN"
        
        # Color code by severity
        def anomaly_severity(ret_val):
            if abs(ret_val) > 0.1: return "🔴"
            elif abs(ret_val) > 0.05: return "🟡"
            else: return "🟢"
        
        recent_anomalies["severity"] = recent_anomalies["ret"].apply(anomaly_severity)
        
        st.markdown("**Weekly price changes** - Large values indicate unusual weekly movements")
        
        st.dataframe(
            recent_anomalies[["severity", "name", "city", "market_type", "date", "price_formatted", "ret_pct"]].assign(
                ret_pct=lambda d: d["ret_pct"].astype(str) + "%"
            ),
            hide_index=True,
            column_config={
                "severity": st.column_config.TextColumn("Alert", width="small"),
                "name": st.column_config.TextColumn("Product", width="medium"),
                "city": st.column_config.TextColumn("City", width="small"),
                "market_type": st.column_config.TextColumn("Market", width="small"),
                "date": st.column_config.DateColumn("Date", width="small"),
                "price_formatted": st.column_config.TextColumn("Price", width="small"),
                "ret_pct": st.column_config.TextColumn("Weekly Change %", width="small")
            }
        )
        
        # Anomaly summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Anomalies", len(recent_anomalies))
        with col2:
            positive_anomalies = len(recent_anomalies[recent_anomalies["ret"] > 0])
            st.metric("Price Spikes", positive_anomalies)
        with col3:
            negative_anomalies = len(recent_anomalies[recent_anomalies["ret"] < 0])
            st.metric("Price Drops", negative_anomalies)
    else:
        st.info("No recent anomalies detected. Market appears stable.")

# Footer
st.divider()
st.markdown("*Data from Bulgarian food price monitoring system*")
