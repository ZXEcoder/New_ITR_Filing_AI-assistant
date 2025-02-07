import streamlit as st
import os
import pandas as pd
import json
from dotenv import load_dotenv
load_dotenv()

# -------------------------------
# Import LangChain Components
# -------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory

# -------------------------------
# Qdrant Vector Database Setup
# -------------------------------
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.docstore.document import Document

# Set Qdrant connection parameters
qdrant_url = os.getenv("QDRANT_URL_LOCALHOST", "http://localhost:6333")
collection_name = "ITR-4"  # This collection should be pre-populated with relevant tax planning docs.

# Initialize embeddings using FastEmbedEmbeddings
embeddings = FastEmbedEmbeddings()
# Initialize Qdrant vector store using QdrantVectorStore.from_existing_collection
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=qdrant_url
)

# -------------------------------
# Tax Calculation Functions
# -------------------------------

def calculate_tax_new_regime(taxable_income):
    """
    Computes tax for the new regime (FY 2025–26 / AY 2026–27) including:
      - Base tax computed as per the slab rates,
      - A rebate component (marginal relief) that reduces the tax liability:
          • For taxable incomes up to ₹12,00,000, the entire computed tax is rebated (net tax = 0).
          • For taxable incomes between ₹12,00,000 and ₹12,75,000, the net tax is capped at 
            (taxable_income - 12,00,000) and the rebate is the difference between the base tax 
            and that excess.
          • For taxable incomes above ₹12,75,000, no rebate is available.
      - Surcharge (if applicable) and Health & Education Cess @ 4% are added.
    
    Returns:
      net_tax: The final tax liability (including surcharge and cess).
      rebate: The computed rebate amount.
    """
    # Compute base tax as per slab rates (without rebate)
    if taxable_income <= 400000:
        base_tax = 0
    elif taxable_income <= 800000:
        base_tax = (taxable_income - 400000) * 0.05
    elif taxable_income <= 1200000:
        base_tax = (800000 - 400000) * 0.05 + (taxable_income - 800000) * 0.10
    elif taxable_income <= 1600000:
        base_tax = ((800000 - 400000) * 0.05 +
                    (1200000 - 800000) * 0.10 +
                    (taxable_income - 1200000) * 0.15)
    elif taxable_income <= 2000000:
        base_tax = ((800000 - 400000) * 0.05 +
                    (1200000 - 800000) * 0.10 +
                    (1600000 - 1200000) * 0.15 +
                    (taxable_income - 1600000) * 0.20)
    elif taxable_income <= 2400000:
        base_tax = ((800000 - 400000) * 0.05 +
                    (1200000 - 800000) * 0.10 +
                    (1600000 - 1200000) * 0.15 +
                    (2000000 - 1600000) * 0.20 +
                    (taxable_income - 2000000) * 0.25)
    else:
        base_tax = ((800000 - 400000) * 0.05 +
                    (1200000 - 800000) * 0.10 +
                    (1600000 - 1200000) * 0.15 +
                    (2000000 - 1600000) * 0.20 +
                    (2400000 - 2000000) * 0.25 +
                    (taxable_income - 2400000) * 0.30)
    
    # Apply rebate based on taxable income:
    if taxable_income <= 1200000:
        rebate = base_tax  # full rebate; net tax becomes 0
        net_tax_base = 0
    elif taxable_income < 1275000:
        net_tax_base = taxable_income - 1200000
        rebate = base_tax - net_tax_base
    else:
        rebate = 0
        net_tax_base = base_tax

    # Surcharge calculation for incomes above ₹50 lakh
    surcharge = 0
    if taxable_income > 5000000:
        if taxable_income <= 10000000:
            surcharge = net_tax_base * 0.10
        elif taxable_income <= 20000000:
            surcharge = net_tax_base * 0.15
        elif taxable_income <= 50000000:
            surcharge = net_tax_base * 0.25
        else:
            surcharge = net_tax_base * 0.25
    net_tax_before_cess = net_tax_base + surcharge

    # Health & Education Cess @ 4%
    cess = net_tax_before_cess * 0.04
    net_tax = net_tax_before_cess + cess

    return net_tax, rebate

def calculate_tax_old_regime(taxable_income, age):
    tax = 0
    if age < 60:
        if taxable_income <= 250000:
            tax = 0
        elif taxable_income <= 500000:
            tax = (taxable_income - 250000) * 0.05
        elif taxable_income <= 1000000:
            tax = 12500 + (taxable_income - 500000) * 0.20
        else:
            tax = 12500 + (1000000 - 500000) * 0.20 + (taxable_income - 1000000) * 0.30
    elif age < 80:
        if taxable_income <= 300000:
            tax = 0
        elif taxable_income <= 500000:
            tax = (taxable_income - 300000) * 0.05
        elif taxable_income <= 1000000:
            tax = 10000 + (taxable_income - 500000) * 0.20
        else:
            tax = 10000 + (1000000 - 500000) * 0.20 + (taxable_income - 1000000) * 0.30
    else:
        if taxable_income <= 500000:
            tax = 0
        elif taxable_income <= 1000000:
            tax = (taxable_income - 500000) * 0.20
        else:
            tax = (1000000 - 500000) * 0.20 + (taxable_income - 1000000) * 0.30

    surcharge = 0
    if taxable_income > 5000000:
        if taxable_income <= 10000000:
            surcharge = tax * 0.10
        elif taxable_income <= 20000000:
            surcharge = tax * 0.15
        elif taxable_income <= 50000000:
            surcharge = tax * 0.25
        else:
            surcharge = tax * 0.37
    tax += surcharge
    cess = tax * 0.04
    tax += cess
    return tax

def calculate_hra_exemption(basic_da, rent_paid, metro_city):
    hra_percent = 0.50 if metro_city else 0.40
    return min(rent_paid - 0.10 * basic_da, hra_percent * basic_da)

# -------------------------------
# LangChain & Google Generative AI Setup with Qdrant Retrieval
# -------------------------------

# We'll use a single input variable "input" for the prompt.
prompt_template = PromptTemplate(
    input_variables=["input"],
    template=(
        "You are an expert tax consultant. Based on the following details:\n\n"
        "{input}\n\n"
        "Provide detailed, legally compliant strategies to minimize the tax liability, ideally reducing it to zero. "
        "Include recommendations for adjustments in deductions, exemptions, or switching tax regimes if beneficial."
    )
)

# Set up conversation memory so that previous queries are remembered.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

tax_planning_chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

# -------------------------------
# Main Application
# -------------------------------

def main():
    st.title("Income Tax Calculator & Tax Planning Advisor")
    
    # Allow user to select between New and Old tax regimes.
    regime = st.selectbox("Select Tax Regime", ["New Regime (Default)", "Old Regime (Optional)"])
    assessment_year = "AY 2026-27" if regime == "New Regime (Default)" else "AY 2025-26"
    st.write(f"Assessment Year: {assessment_year}")
    
    st.header("Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        residential_status = st.selectbox("Residential Status", ["Resident", "Non-Resident"])
    with col2:
        city_type = st.radio("City Type", ["Metro City", "Non-Metro City"])
    
    st.header("Income Details")
    salary = st.number_input("Annual Salary (₹)", min_value=0.0, value=0.0)
    business_income = st.number_input("Business Income (₹)", min_value=0.0, value=0.0)
    savings_interest = st.number_input("Savings A/C Interest (₹)", min_value=0.0, value=0.0)
    other_interest = st.number_input("Other Interest Income (₹)", min_value=0.0, value=0.0)
    rental_income = st.number_input("Annual Rental Income (₹)", min_value=0.0, value=0.0)
    
    if regime == "Old Regime (Optional)":
        st.subheader("Rental Property Deductions")
        house_loan_interest = st.number_input("Interest paid on loans for houses on rent (₹)", min_value=0.0, value=0.0)
        municipal_taxes = st.number_input("Municipal taxes paid on houses on rent (₹)", min_value=0.0, value=0.0)
    else:
        house_loan_interest = 0.0
        municipal_taxes = 0.0

    # -------------------------------
    # Deductions Section
    # -------------------------------
    st.header("Deductions")
    if regime == "New Regime (Default)":
        st.info("Under the New Tax Regime, most exemptions (80C, HRA, 80D, etc.) are not allowed. Only a standard deduction of ₹75,000 and employer's NPS contribution are permitted.")
        nps_80ccd_2 = st.number_input("Employer's NPS Contribution (80CCD(2)) (₹)", min_value=0.0, value=0.0)
        total_deductions = 75000 + nps_80ccd_2
        hra_exemption = 0
    else:
        st.info("Under the Old Tax Regime, you can claim various deductions and exemptions.")
        sec_80c = st.number_input("Tax Saving Investment (80C) (₹)", min_value=0.0, max_value=150000.0, value=0.0)
        home_loan_interest_self = st.number_input("Interest paid on self-occupied home loan (₹)", min_value=0.0, value=0.0)
        education_loan_interest = st.number_input("Interest paid on Educational Loan (₹)", min_value=0.0, value=0.0)
        
        st.subheader("HRA Details (if applicable)")
        rent_paid = st.number_input("Annual Rent Paid (₹)", min_value=0.0, value=0.0)
        basic_da = st.number_input("Annual Basic + DA (₹)", min_value=0.0, value=0.0)
        hra_exemption = calculate_hra_exemption(basic_da, rent_paid, city_type=="Metro City") if (rent_paid > 0 and basic_da > 0) else 0
        
        st.subheader("Health Insurance")
        health_insurance_self = st.number_input("Health Insurance (Self) (₹)", min_value=0.0, max_value=25000.0, value=0.0)
        health_insurance_parents = st.number_input("Health Insurance (Parents) (₹)", min_value=0.0, max_value=50000.0, value=0.0)
        
        st.subheader("NPS Contributions")
        nps_80ccd_1 = st.number_input("NPS Contribution (80CCD(1)) (₹)", min_value=0.0, max_value=50000.0, value=0.0)
        nps_80ccd_2 = st.number_input("Employer's NPS Contribution (80CCD(2)) (₹)", min_value=0.0, value=0.0)
        
        other_deductions = st.number_input("Other Deductions (₹)", min_value=0.0, value=0.0)
        
        total_deductions = (
            min(sec_80c, 150000) +
            min(health_insurance_self, 25000) +
            min(health_insurance_parents, 50000) +
            min(nps_80ccd_1, 50000) +
            nps_80ccd_2 +
            home_loan_interest_self +
            education_loan_interest +
            other_deductions
        )
    
    # -------------------------------
    # Calculate Total Income
    # -------------------------------
    gross_salary = salary - (hra_exemption if regime=="Old Regime (Optional)" else 0)
    rental_net = rental_income - house_loan_interest - municipal_taxes
    total_income = gross_salary + business_income + savings_interest + other_interest + rental_net
    
    # -------------------------------
    # Taxable Income and Tax Calculation
    # -------------------------------
    taxable_income = max(0, total_income - total_deductions)
    
    if regime == "New Regime (Default)":
        tax, rebate = calculate_tax_new_regime(taxable_income)
    else:
        tax = calculate_tax_old_regime(taxable_income, age)
        rebate = None  # Not applicable in the old regime
    
    # -------------------------------
    # Display Tax Calculation Results
    # -------------------------------
    st.header("Tax Calculation Results")
    results_data = {
        'Component': ['Total Income', 'Total Deductions', 'Taxable Income', 'Income Tax (pre-cess)', 'Cess (4%)', 'Total Tax Liability'],
        'Amount (₹)': [
            total_income,
            total_deductions,
            taxable_income,
            tax - (tax * 0.04),
            tax * 0.04,
            tax
        ]
    }
    if regime == "New Regime (Default)":
        results_data['Component'].append('Rebate')
        results_data['Amount (₹)'].append(rebate)
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df.style.format({'Amount (₹)': '{:,.2f}'}))
    
    # Display Tax Slabs Information
    st.subheader("Tax Slabs")
    if regime == "New Regime (Default)":
        slabs = {
            'Income Range (₹)': [
                'Up to ₹4,00,000',
                '₹4,00,001 to ₹8,00,000',
                '₹8,00,001 to ₹12,00,000',
                '₹12,00,001 to ₹16,00,000',
                '₹16,00,001 to ₹20,00,000',
                '₹20,00,001 to ₹24,00,000',
                'Above ₹24,00,000'
            ],
            'Tax Rate': ['Nil', '5%', '10%', '15%', '20%', '25%', '30%']
        }
    else:
        slabs = {
            'Income Range (₹)': [
                'Up to ₹2,50,000',
                '₹2,50,001 to ₹5,00,000',
                '₹5,00,001 to ₹10,00,000',
                'Above ₹10,00,000'
            ],
            'Tax Rate': ['Nil', '5%', '20%', '30%']
        }
    slabs_df = pd.DataFrame(slabs)
    st.table(slabs_df)
    
    # -------------------------------
    # Prepare Tax Data for Retrieval-Augmented LLM Query
    # -------------------------------
    tax_summary = {
        "total_income": total_income,
        "total_deductions": total_deductions,
        "taxable_income": taxable_income,
        "tax_liability": tax
    }
    tax_summary_json = json.dumps(tax_summary, indent=2)
    
    query_for_docs = "Tax planning advice for these tax details: " + tax_summary_json
    retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(query_for_docs)
    retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Show document similarity search results
    with st.expander("View Similar Documents Retrieved"):
        for i, doc in enumerate(retrieved_docs, start=1):
            st.write(f"Document {i}:")
            st.info(doc.page_content)
    
    # -------------------------------
    # LLM Query Section with Memory
    # -------------------------------
    st.header("Tax Planning Advice Query")
    user_query = st.text_input("Enter your query:", placeholder="Ask your question about your tax situation...")
    
    if user_query:
        # Combine tax data, retrieved context, and user query into one input string.
        full_input = (
            f"Tax Data:\n{tax_summary_json}\n\n"
            f"Retrieved Documents:\n{retrieved_context}\n\n"
            f"User Query: {user_query}"
        )
        # Run the LLM chain with the combined input.
        llm_response = tax_planning_chain.run(input=full_input)
        st.subheader("LLM Advice Response")
        st.write(llm_response)
    
if __name__ == "__main__":
    main()
