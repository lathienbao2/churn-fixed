"""
Churn Prediction System v2.0
Giao diện Streamlit cải tiến: charts, UX tốt hơn, đề xuất giải pháp chi tiết
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import json
from io import StringIO
import traceback

warnings.filterwarnings('ignore')

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ — Dự Đoán Khách Hàng Rời Đi",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS tùy chỉnh ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }

/* Header gradient */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99,179,237,0.2);
}
.main-header h1 { color: #e2e8f0; margin:0; font-size:1.8rem; font-weight:700; }
.main-header p  { color: #94a3b8; margin:.4rem 0 0; font-size:.95rem; }

/* Risk badge */
.risk-high   { background:#fee2e2; color:#991b1b; padding:.3rem .9rem; border-radius:20px; font-weight:600; font-size:.85rem; }
.risk-medium { background:#fef3c7; color:#92400e; padding:.3rem .9rem; border-radius:20px; font-weight:600; font-size:.85rem; }
.risk-low    { background:#d1fae5; color:#065f46; padding:.3rem .9rem; border-radius:20px; font-weight:600; font-size:.85rem; }

/* Rec cards */
.rec-card {
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: .85rem 1rem;
    margin: .5rem 0;
}
.rec-card.high  { border-color: #ef4444; }
.rec-card.medium{ border-color: #f59e0b; }
.rec-card.low   { border-color: #10b981; }
.rec-title { font-weight:600; color:#1e293b; font-size:.95rem; }
.rec-detail { color:#475569; font-size:.88rem; margin-top:.25rem; }
.rec-meta   { display:flex; gap:.75rem; margin-top:.4rem; }
.tag { font-size:.75rem; padding:.15rem .5rem; border-radius:10px; background:#e2e8f0; color:#475569; }

/* Metric override */
[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: .75rem 1rem !important;
}

/* Gauge */
.gauge-wrap { text-align:center; }
.gauge-num  { font-size:3rem; font-weight:700; }
.gauge-high   { color:#ef4444; }
.gauge-medium { color:#f59e0b; }
.gauge-low    { color:#10b981; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label { color: #94a3b8 !important; font-size:.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Import config ────────────────────────────────────────────────────────────
from datasets_config import DATASET_CONFIGS, detect_dataset_type, get_retention_recommendations
from churniq.modeling import metric_explanations_vi
from churniq.monitoring import drift_report, save_drift_report
from churniq.prediction import feature_importances, predict_batch as shared_predict_batch, predict_single as shared_predict_single

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()

# ─── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải models...")
def load_models():
    models = {}
    for key, config in DATASET_CONFIGS.items():
        path = config['model_path']
        if os.path.exists(path):
            try:
                models[key] = {'model': joblib.load(path), 'config': config}
            except Exception as e:
                st.warning(f"⚠️ Không load được model {key}: {e}")
    return models

models_dict = load_models()

# ─── Helper functions ─────────────────────────────────────────────────────────
def get_feature_importances(model_entry, input_df):
    """Trả về dict {tên feature gốc: importance}"""
    return feature_importances(model_entry)

def run_prediction(model_entry, input_df, threshold=0.5):
    model = model_entry['model']
    config = model_entry['config']
    feature_cols = config['numeric_features'] + config['categorical_features']
    result = shared_predict_single(model_entry, input_df[feature_cols], threshold=threshold)
    return result['churn_probability'], result['is_churn']

def risk_badge(prob):
    if prob > 0.7:
        return '<span class="risk-high">🔴 Nguy cơ CAO</span>'
    elif prob > 0.4:
        return '<span class="risk-medium">🟡 Nguy cơ TRUNG BÌNH</span>'
    return '<span class="risk-low">🟢 Nguy cơ THẤP</span>'

def render_gauge(prob):
    pct = prob * 100
    cls = "gauge-high" if prob > 0.7 else ("gauge-medium" if prob > 0.4 else "gauge-low")
    return f"""
    <div class="gauge-wrap">
        <div class="gauge-num {cls}">{pct:.1f}%</div>
        <div style="color:#64748b;font-size:.9rem;margin-top:.25rem;">Xác suất rời đi</div>
    </div>"""

def render_recs(recs_data):
    priority = recs_data['priority']
    recs = recs_data['recommendations']
    impact_color = {'Rất cao': 'high', 'Cao': 'high', 'Trung bình': 'medium', 'Thấp': 'low', 'Rất thấp': 'low'}
    html = ""
    for r in recs:
        cls = impact_color.get(r.get('impact', 'Thấp'), 'low')
        html += f"""
        <div class="rec-card {cls}">
            <div class="rec-title">{r['action']}</div>
            <div class="rec-detail">{r['detail']}</div>
            <div class="rec-meta">
                <span class="tag">Impact: {r.get('impact','?')}</span>
                <span class="tag">Effort: {r.get('effort','?')}</span>
            </div>
        </div>"""
    return html


# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🔮 ChurnIQ — Hệ Thống Dự Đoán Khách Hàng Rời Đi</h1>
    <p>Phân tích nguy cơ churn & đề xuất giải pháp giữ chân khách hàng theo thời gian thực</p>
</div>
""", unsafe_allow_html=True)

if not models_dict:
    st.error("❌ Không tìm thấy model nào. Vui lòng chạy `python train_model.py` trước.")
    st.stop()

# ─── Model status bar ────────────────────────────────────────────────────────
cols_status = st.columns(len(models_dict) + 1)
with cols_status[0]:
    st.markdown(f"**{len(models_dict)} models sẵn sàng**")
for i, (key, info) in enumerate(models_dict.items()):
    with cols_status[i + 1]:
        st.success(f"✅ {info['config']['name']}")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab_single, tab_batch, tab_insight = st.tabs([
    "🎯 Dự Đoán Đơn Lẻ",
    "📦 Dự Đoán Hàng Loạt",
    "📊 Phân Tích & Insights"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: DỰ ĐOÁN ĐƠN LẺ
# ─────────────────────────────────────────────────────────────────────────────
with tab_single:
    col_form, col_result = st.columns([1, 1.2], gap="large")

    with col_form:
        st.subheader("📝 Nhập thông tin khách hàng")

        selected_type = st.selectbox(
            "Loại dataset",
            options=list(models_dict.keys()),
            format_func=lambda x: DATASET_CONFIGS[x]['name'],
            key="single_type"
        )
        threshold = st.slider(
            "Ngưỡng quyết định churn",
            min_value=0.30,
            max_value=0.70,
            value=0.50,
            step=0.05,
            help="Giảm ngưỡng để tăng Recall; tăng ngưỡng để tăng Precision."
        )
        config = models_dict[selected_type]['config']
        labels = config.get('feature_labels', {})
        options_map = config.get('categorical_options', {})
        defaults_num = config.get('numeric_defaults', {})

        inputs = {}
        with st.container():
            st.markdown("**📐 Thông số số**")
            ncols = st.columns(min(3, len(config['numeric_features'])))
            for i, feat in enumerate(config['numeric_features']):
                with ncols[i % len(ncols)]:
                    label = labels.get(feat, feat)
                    default = defaults_num.get(feat, 0.0)
                    step = 0.1 if any(k in feat.lower() for k in ['charges', 'minutes']) else 1.0
                    inputs[feat] = st.number_input(label, value=default, step=step, key=f"num_{feat}")

        with st.container():
            st.markdown("**🏷️ Thông số phân loại**")
            ccols = st.columns(min(2, len(config['categorical_features'])))
            for i, feat in enumerate(config['categorical_features']):
                with ccols[i % len(ccols)]:
                    label = labels.get(feat, feat)
                    opts = options_map.get(feat, ['No', 'Yes'])
                    inputs[feat] = st.selectbox(label, opts, key=f"cat_{feat}")

        predict_btn = st.button("🔮 Dự Đoán Ngay", type="primary", use_container_width=True)

    with col_result:
        st.subheader("📊 Kết Quả Phân Tích")

        if predict_btn:
            try:
                model_entry = models_dict[selected_type]
                input_df = pd.DataFrame([inputs])
                for feat in config['numeric_features']:
                    input_df[feat] = pd.to_numeric(input_df[feat], errors='coerce')

                churn_prob, is_churn = run_prediction(model_entry, input_df, threshold=threshold)
                recs_data = get_retention_recommendations(selected_type, inputs, churn_prob)

                # Gauge
                st.markdown(render_gauge(churn_prob), unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center;margin:.5rem 0'>{risk_badge(churn_prob)}</div>", unsafe_allow_html=True)
                st.markdown("")

                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("Dự đoán", "⚠️ Rời đi" if is_churn else "✅ Ở lại")
                m2.metric("Xác suất", f"{churn_prob*100:.1f}%")
                m3.metric("Ngưỡng", f"{threshold:.2f}")

                # Feature importance bar chart
                fi = get_feature_importances(model_entry, input_df)
                if fi:
                    st.markdown("**🔍 Yếu tố ảnh hưởng nhiều nhất**")
                    # Top 6 features
                    top_fi = dict(sorted(fi.items(), key=lambda x: -x[1])[:6])
                    fi_df = pd.DataFrame({'Feature': list(top_fi.keys()), 'Importance': list(top_fi.values())})
                    fi_df = fi_df.sort_values('Importance')
                    st.bar_chart(fi_df.set_index('Feature'))

                # Recommendations
                st.markdown("**💡 Đề xuất giải pháp giữ chân**")
                st.markdown(render_recs(recs_data), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Lỗi dự đoán: {e}")
                with st.expander("Chi tiết lỗi"):
                    st.code(traceback.format_exc())
        else:
            st.info("👈 Điền thông tin khách hàng và nhấn **Dự Đoán Ngay**")
            st.markdown("""
            **Hướng dẫn sử dụng:**
            - Chọn loại dataset phù hợp với dữ liệu của bạn
            - Nhập thông số khách hàng cần kiểm tra
            - Xem kết quả xác suất churn & đề xuất giải pháp
            """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: DỰ ĐOÁN HÀNG LOẠT
# ─────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("📤 Tải lên file CSV để dự đoán hàng loạt")

    col_up, col_fmt = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Kéo thả file CSV vào đây", type=["csv"], label_visibility="collapsed")

    with col_fmt:
        with st.expander("📖 Định dạng CSV hỗ trợ"):
            st.markdown("""
**Telco IBM:** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `PaymentMethod`, `InternetService`, `TechSupport`, `OnlineSecurity`

**Call Details:** `Account length`, `Total day minutes`, `Total eve minutes`, `Total night minutes`, `Total intl minutes`, `Number vmail messages`, `Customer service calls`, `International plan`, `Voice mail plan`
            """)

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            st.markdown(f"**Preview:** {len(df_raw):,} dòng × {len(df_raw.columns)} cột")
            st.dataframe(df_raw.head(3), use_container_width=True, hide_index=True)

            dataset_type = detect_dataset_type(df_raw)
            if not dataset_type:
                st.error("❌ Không nhận diện được định dạng dataset!")
                st.stop()

            st.success(f"✅ Phát hiện: **{DATASET_CONFIGS[dataset_type]['name']}**")
            batch_threshold = st.slider(
                "Ngưỡng quyết định batch",
                min_value=0.30,
                max_value=0.70,
                value=0.50,
                step=0.05,
                key="batch_threshold"
            )

            if dataset_type not in models_dict:
                st.error("❌ Chưa có model cho dataset này!")
                st.stop()

            if st.button("🚀 Bắt đầu dự đoán hàng loạt", type="primary", use_container_width=True):
                with st.spinner("Đang xử lý..."):
                    model_entry = models_dict[dataset_type]
                    config = model_entry['config']
                    model = model_entry['model']

                    df = df_raw.copy()
                    rename_map = config.get('column_rename', {})
                    df = df.rename(columns=rename_map)

                    for col in config['numeric_features']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    if 'TotalCharges' in df.columns:
                        dropped = df['TotalCharges'].isna().sum()
                        df = df.dropna(subset=['TotalCharges'])
                        if dropped:
                            st.warning(f"⚠️ Bỏ {dropped} dòng do TotalCharges không hợp lệ")

                    feature_cols = config['numeric_features'] + config['categorical_features']
                    missing = [c for c in feature_cols if c not in df.columns]
                    if missing:
                        st.error(f"❌ File thiếu cột: {missing}")
                        st.stop()

                    df = shared_predict_batch(model_entry, df[feature_cols], threshold=batch_threshold)

                    total = len(df)
                    if total == 0:
                        st.error("❌ Không còn dòng hợp lệ sau khi làm sạch dữ liệu")
                        st.stop()
                    churn_cnt = int(df["Churn_Prediction"].sum())
                    high_risk = int((df["Churn_Probability"] > 0.75).sum())
                    med_risk  = int(((df["Churn_Probability"] > 0.4) & (df["Churn_Probability"] <= 0.75)).sum())
                    low_risk  = total - high_risk - med_risk

                # ── Summary metrics ──
                st.markdown("### 📊 Tổng Quan Kết Quả")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Tổng khách hàng", f"{total:,}")
                c2.metric("Dự đoán rời đi", f"{churn_cnt:,}", f"{churn_cnt/total*100:.1f}%")
                c3.metric("🔴 Nguy cơ cao (>75%)", f"{high_risk:,}")
                c4.metric("🟡 Nguy cơ TB (40-75%)", f"{med_risk:,}")

                # ── Charts ──
                chart_c1, chart_c2 = st.columns(2)

                with chart_c1:
                    st.markdown("**Phân phối xác suất churn**")
                    hist_data = pd.cut(
                        df["Churn_Probability"],
                        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
                    ).value_counts().sort_index()
                    st.bar_chart(hist_data)

                with chart_c2:
                    st.markdown("**Phân tầng rủi ro**")
                    risk_df = pd.DataFrame({
                        'Mức độ rủi ro': ['🔴 Cao (>75%)', '🟡 Trung bình (40-75%)', '🟢 Thấp (<40%)'],
                        'Số khách hàng': [high_risk, med_risk, low_risk]
                    })
                    st.bar_chart(risk_df.set_index('Mức độ rủi ro'))

                # Drift warning against bundled training sample
                sample_path = os.path.join(BASE_DIR, 'data',
                    'Telco_customer_churn.csv' if dataset_type == 'telco_ibm' else 'Churn.csv')
                if os.path.exists(sample_path):
                    train_sample = pd.read_csv(sample_path).rename(columns=config.get('column_rename', {}))
                    drift = drift_report(train_sample, df, config, threshold=0.2)
                    report_path = save_drift_report(drift, os.path.join(BASE_DIR, 'logs'))
                    if drift['drift_detected']:
                        st.warning(f"⚠️ Drift detected: PSI cao nhất {drift['max_psi']:.3f}. Report: {report_path}")
                    else:
                        st.info(f"PSI ổn định: cao nhất {drift['max_psi']:.3f}")

                # ── High-risk table ──
                if high_risk > 0:
                    st.markdown(f"### 🚨 Top {min(20, high_risk)} Khách Hàng Nguy Cơ Cao Nhất")
                    high_df = df.nlargest(min(20, high_risk), "Churn_Probability")

                    id_col = next((c for c in ['customerID', 'CustomerID', 'Account length', 'State'] if c in high_df.columns), None)
                    show_cols = ([id_col] if id_col else []) + feature_cols[:4] + ["Churn_Probability"]
                    show_cols = [c for c in show_cols if c in high_df.columns]

                    st.dataframe(
                        high_df[show_cols].style.format({"Churn_Probability": "{:.1%}"})
                            .background_gradient(subset=["Churn_Probability"], cmap="RdYlGn_r"),
                        use_container_width=True,
                        hide_index=True
                    )

                # ── Download ──
                csv_out = df.to_csv(index=False)
                st.download_button(
                    "💾 Tải kết quả đầy đủ (CSV)",
                    data=csv_out,
                    file_name=f"churn_predictions_{dataset_type}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
            with st.expander("Chi tiết lỗi"):
                st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: PHÂN TÍCH & INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_insight:
    st.subheader("📊 Phân Tích Dữ Liệu Mẫu & Feature Importance")

    ins_type = st.selectbox(
        "Chọn dataset để phân tích",
        list(models_dict.keys()),
        format_func=lambda x: DATASET_CONFIGS[x]['name'],
        key="insight_type"
    )

    config = models_dict[ins_type]['config']
    data_path = os.path.join(BASE_DIR, 'data',
        'Telco_customer_churn.csv' if ins_type == 'telco_ibm' else 'Churn.csv')

    if os.path.exists(data_path):
        @st.cache_data
        def load_sample(path, dtype):
            df = pd.read_csv(path)
            if dtype == 'call_details':
                rename_map = DATASET_CONFIGS['call_details'].get('column_rename', {})
                df = df.rename(columns=rename_map)
            return df

        df_sample = load_sample(data_path, ins_type)

        # Detect target column
        target_col = 'Churn'
        if target_col in df_sample.columns:
            df_sample[target_col] = df_sample[target_col].astype(str).str.lower().map(
                {'yes': 1, 'no': 0, 'true.': 1, 'false.': 0, 'true': 1, 'false': 0, '1': 1, '0': 0}
            ).fillna(0)

        st.markdown(f"**Dataset:** {len(df_sample):,} records | {len(df_sample.columns)} cột")

        # ── Churn rate overview ──
        if target_col in df_sample.columns:
            churn_rate = df_sample[target_col].mean()
            c1, c2, c3 = st.columns(3)
            c1.metric("Tỷ lệ churn thực tế", f"{churn_rate*100:.1f}%")
            c2.metric("Số khách hàng churn", f"{int(df_sample[target_col].sum()):,}")
            c3.metric("Số khách hàng ở lại", f"{int((df_sample[target_col]==0).sum()):,}")

        # ── Feature Importance từ model ──
        st.markdown("### 🔍 Tầm Quan Trọng Của Từng Đặc Trưng (Model)")
        model_obj = models_dict[ins_type]['model']
        try:
            clf = model_obj.named_steps['classifier']
            pre = model_obj.named_steps['preprocessor']
            fn = pre.get_feature_names_out()
            imp = clf.feature_importances_

            fi_df = pd.DataFrame({'Feature': fn, 'Importance': imp})
            fi_df['Feature_Clean'] = fi_df['Feature'].str.split('__').str[-1]
            fi_df = fi_df.sort_values('Importance', ascending=False).head(12)

            st.bar_chart(fi_df.set_index('Feature_Clean')['Importance'])

            st.markdown("**Top yếu tố ảnh hưởng đến churn:**")
            for _, row in fi_df.head(5).iterrows():
                label = config.get('feature_labels', {}).get(row['Feature_Clean'].split('_')[0], row['Feature_Clean'])
                bar = "█" * int(row['Importance'] * 100)
                st.markdown(f"`{bar:<20}` **{row['Feature_Clean']}** — {row['Importance']*100:.1f}%")
        except Exception as e:
            st.warning(f"Không lấy được feature importance: {e}")

        metrics_path = models_dict[ins_type]['config']['model_path'].replace('.pkl', '_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            st.markdown("### 📋 Bronze Metrics & Classification Report")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("AUC", f"{metrics_data.get('test_auc', 0):.3f}")
            m2.metric("Precision", f"{metrics_data.get('test_precision', 0):.3f}")
            m3.metric("Recall", f"{metrics_data.get('test_recall', 0):.3f}")
            m4.metric("F1", f"{metrics_data.get('test_f1', 0):.3f}")
            explanations = metric_explanations_vi()
            for name, text in explanations.items():
                st.caption(f"**{name}:** {text}")
            if metrics_data.get('confusion_matrix'):
                st.write("Confusion matrix", metrics_data['confusion_matrix'])
            if metrics_data.get('classification_report'):
                st.code(metrics_data['classification_report'])
            if metrics_data.get('threshold_metrics'):
                st.dataframe(pd.DataFrame(metrics_data['threshold_metrics']).T, use_container_width=True)

        # ── Distribution charts ──
        if target_col in df_sample.columns:
            st.markdown("### 📈 Phân Tích Đặc Trưng Theo Churn")
            num_feats = config['numeric_features']
            cols = st.columns(min(3, len(num_feats)))
            for i, feat in enumerate(num_feats[:3]):
                if feat in df_sample.columns:
                    with cols[i]:
                        st.markdown(f"**{feat}**")
                        numeric_series = pd.to_numeric(df_sample[feat], errors='coerce')
                        grp = (
                            df_sample.assign(**{feat: numeric_series})
                            .groupby(target_col, observed=False)[feat]
                            .mean()
                            .dropna()
                            .reset_index()
                        )
                        if grp.empty:
                            st.info(f"Không có dữ liệu số hợp lệ cho {feat}")
                        else:
                            grp[target_col] = grp[target_col].map({0: 'Ở lại', 1: 'Rời đi'})
                            st.bar_chart(grp.set_index(target_col)[feat])

            # Categorical breakdown
            cat_feats = config['categorical_features']
            if cat_feats:
                st.markdown("### 📊 Tỷ Lệ Churn Theo Đặc Trưng Phân Loại")
                cat_cols = st.columns(min(2, len(cat_feats)))
                for i, feat in enumerate(cat_feats[:4]):
                    if feat in df_sample.columns:
                        with cat_cols[i % 2]:
                            st.markdown(f"**{feat}**")
                            grp = df_sample.groupby(feat)[target_col].mean().sort_values(ascending=False)
                            grp_df = grp.reset_index()
                            grp_df.columns = [feat, 'Churn Rate']
                            grp_df['Churn Rate'] = grp_df['Churn Rate'] * 100
                            st.bar_chart(grp_df.set_index(feat)['Churn Rate'])

    else:
        st.warning(f"Không tìm thấy file dữ liệu mẫu tại: {data_path}")
        st.info("Đặt file CSV vào thư mục `data/` để xem phân tích.")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-size:.8rem'>"
    "🔮 ChurnIQ v2.0 | Powered by RandomForest + Streamlit | "
    "REST API: <code>python api.py</code>"
    "</div>",
    unsafe_allow_html=True
)
