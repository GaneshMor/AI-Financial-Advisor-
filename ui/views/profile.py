import streamlit as st

from ui import state
from ui.components import disclaimer, page_header
from ui.nav import PAGES
from utils.validation import business_warnings, validate_inputs

DEFAULT_PROFILE = dict(age=30, monthly_income=80000.0, monthly_expenses=40000.0, monthly_debt=0.0,
                       current_savings=100000.0, current_investments=100000.0, monthly_investment=10000.0,
                       dependents=0, emergency_fund=150000.0)


def render():
    page_header("User Profile", "Monthly figures in rupees. All amounts must be 0 or more.")
    d = st.session_state.profile_data or DEFAULT_PROFILE

    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", min_value=18, max_value=100, value=int(d["age"]), step=1)
        dependents = c2.number_input("Number of dependents", min_value=0, max_value=20, value=int(d["dependents"]), step=1)
        c1, c2, c3 = st.columns(3)
        income = c1.number_input("Monthly income (₹)", min_value=0.0, value=float(d["monthly_income"]), step=1000.0)
        expenses = c2.number_input("Monthly living expenses (₹)", min_value=0.0, value=float(d["monthly_expenses"]),
                                   step=1000.0, help="Rent, food, bills, etc. Do NOT include EMIs or investments.")
        debt = c3.number_input("Monthly debt / EMIs (₹)", min_value=0.0, value=float(d["monthly_debt"]), step=500.0)
        c1, c2, c3 = st.columns(3)
        investment = c1.number_input("Current monthly investment / SIP (₹)", min_value=0.0,
                                     value=float(d["monthly_investment"]), step=500.0)
        savings = c2.number_input("Current savings (₹)", min_value=0.0, value=float(d["current_savings"]),
                                  step=10000.0, help="Bank and cash savings, excluding the emergency fund.")
        investments = c3.number_input("Current investments (₹)", min_value=0.0, value=float(d["current_investments"]),
                                      step=10000.0, help="Mutual funds, shares, PPF and similar.")
        emergency = st.number_input("Emergency fund (₹)", min_value=0.0, value=float(d["emergency_fund"]), step=10000.0,
                                    help="Only money kept aside for emergencies.")
        submitted = st.form_submit_button("Save and continue", type="primary")

    if submitted:
        data = dict(age=int(age), monthly_income=income, monthly_expenses=expenses, monthly_debt=debt,
                    current_savings=savings, current_investments=investments, monthly_investment=investment,
                    dependents=int(dependents), emergency_fund=emergency)
        report = validate_inputs(data)
        if not report.ok:
            for e in report.errors:
                st.error(e)
        else:
            st.session_state.profile_data = data
            st.success("Profile saved.")

    if st.session_state.profile_data:
        report = validate_inputs(st.session_state.profile_data)
        for w in business_warnings(report.profile, None):
            st.warning(w)
        if st.button("Next: Financial Goal", icon=":material/arrow_forward:"):
            st.switch_page(PAGES["goal"])
    disclaimer()
