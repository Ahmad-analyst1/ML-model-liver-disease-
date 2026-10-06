import streamlit as st
from app.predictor import predict

st.set_page_config(page_title="Liver Screening", page_icon="🩺")
st.title("🩺 Liver Disease Screening")
st.caption("Research prototype based on ILPD + Logistic Regression. Not a diagnostic tool.")
c1, c2 = st.columns(2)
age = c1.number_input("Age", 1, 120, 45)
gender = c2.selectbox("Sex", ["Male", "Female"])
tb = c1.number_input("Total bilirubin (mg/dL)", 0.0, 100.0, 1.0)
db = c2.number_input("Direct bilirubin (mg/dL)", 0.0, 50.0, 0.3)
alp = c1.number_input("Alkaline phosphatase (IU/L)", 10.0, 5000.0, 200.0)
alt = c2.number_input("ALT / SGPT (IU/L)", 1.0, 5000.0, 35.0)
ast = c1.number_input("AST / SGOT (IU/L)", 1.0, 10000.0, 40.0)
tp = c2.number_input("Total proteins (g/dL)", 1.0, 15.0, 6.8)
alb = c1.number_input("Albumin (g/dL)", 0.5, 8.0, 3.5)
ag = c2.number_input("Albumin/Globulin ratio", 0.1, 5.0, 1.0)
if st.button("Assess", type="primary"):
    r = predict(dict(age=age, gender=int(gender == "Male"), total_bilirubin=tb, direct_bilirubin=db,
                     alk_phosphatase=alp, alt=alt, ast=ast, total_proteins=tp, albumin=alb, ag_ratio=ag))
    (st.error if r["flag"] else st.success)(f"Risk: {r['risk_band'].upper()}  |  probability {r['probability']:.0%}")
    st.caption(r["disclaimer"])
