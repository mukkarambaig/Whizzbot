from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

examples = [
    {
        "input": "What are the guidelines for overtime pay?",
        "context": "The provided document states that employees are eligible for overtime pay at a rate of 1.5 times their regular hourly rate for hours worked beyond 40 in a week.",
        "output": '''
        - Employees receive overtime pay at 1.5 times the regular rate.
        - Overtime is applicable after 40 hours of work in a week.'''
    },
    {
        "input": "How does the company support employee professional development?",
        "context": "The document mentions that the company offers an annual allowance of $1000 for professional development courses for each employee.",
        "output": '''
        - The company provides a $1000 annual allowance for professional development.
        - This allowance can be used for courses to enhance professional skills.'''
    },
    {
        "input": "Can employees work remotely from another country?",
        "context": "There is no information about remote work policies from another country in the provided document.",
        "output": "- I apologize, but the provided document does not contain information on remote work from another country. Please provide a relevant document or context."
    },
    {
        "input": "What is the company's policy on sick leave?",
        "context": "The document outlines that employees are entitled to 10 days of paid sick leave per year.",
        "output": "- Employees are entitled to 10 days of paid sick leave annually."
    },
    {
        "input": "Describe the company's employee wellness program.",
        "context": "The provided document highlights a wellness program that includes free gym memberships, monthly wellness workshops, and annual health screenings.",
        "output": '''
        - Free gym memberships provided.
        - Monthly wellness workshops available.
        - Annual health screenings offered.'''
    },
    {
        "input": "What are the eligibility criteria for the company's tuition reimbursement program?",
        "context": "Employees must have been with the company for at least one year and maintain a grade of B or higher in the course to be eligible for tuition reimbursement.",
        "output": '''
        - Must be with the company for at least one year.
        - Must maintain a grade of B or higher for reimbursement.'''
    },
    {
        "input": "How does the company address grievances?",
        "context": "The document details a step-by-step grievance procedure starting with a written complaint to the HR department, followed by a meeting with the involved parties, and a resolution within 30 days.",
        "output": '''
        - Start with a written complaint to HR.
        - A meeting with involved parties will be held.
        - Resolution provided within 30 days.'''
    },
    {
        "input": "What are the guidelines for remote work?",
        "context": "The document specifies that remote work must be approved by the direct manager and that employees must be available during core business hours.",
        "output": '''
        - Approval by direct manager required.
        - Must be available during core business hours.'''
    },
    {
        "input": "Are there any company-provided child care benefits?",
        "context": "The document does not mention any child care benefits provided by the company.",
        "output": "- I apologize, but the provided document does not contain information on child care benefits. Please provide a relevant document or context."
    },
    {
        "input": "What is the policy on employee referrals?",
        "context": "The document states that employees receive a bonus of $500 for each successful referral after the new hire completes three months of employment.",
        "output": '''
        - $500 bonus for each successful referral.
        - Bonus awarded after new hire completes three months.'''
    },
    {
        "input": "How is employee performance evaluated?",
        "context": "Performance evaluations are conducted annually, based on key performance indicators and feedback from supervisors and peers.",
        "output": '''
        - Annual evaluations based on key performance indicators.
        - Includes feedback from supervisors and peers.'''
    },
    {
        "input": "What is the protocol for requesting annual leave?",
        "context": "Employees must submit a leave request form at least one month in advance, and approval is subject to the department's staffing needs.",
        "output": '''
        - Submit leave request one month in advance.
        - Approval depends on department's staffing needs.'''
    },
    {
        "input": "What support does the company offer for relocating employees?",
        "context": "The document mentions a relocation package that includes moving expenses, a temporary housing allowance for three months, and assistance in finding permanent housing.",
        "output": '''
        - Covers moving expenses.
        - Provides a temporary housing allowance for three months.
        - Assists in finding permanent housing.'''
    },
]

def prompt_generator(prompt: str, context: str,) -> str:
    prompt_template = PromptTemplate(
    input_variables=["input", "context", "output"],
    template="input: {input}\n\ncontext: {context}\n\noutput: {output}"
    )
    
    example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=prompt_template,
    max_length=2000
    )
    
    dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=prompt_template,
    prefix="""You are an advanced AI assistant designed to function as an HR manager, with complete knowledge of a specific provided document. Your task is to provide clear and understandable answers to queries based on this document's content. Adhere to these guidelines when responding:

         - Rely STRICTLY solely on the information from the provided document to answer questions. If the document does not contain the necessary information, apologize and clarify that the relevant context is missing.
         - Keep your answers concise, using no more than FIVE sentences ONLY!
         - Format your answers in BULLET POINTS to ensure they are easy to read and understand, even for non-English speakers.
         
         Use the following examples to guide your responses:
         """,
    suffix="input: {input}\n\ncontext: {context}\n\noutput:",
    input_variables=["input", "context"],
    )
    
    return dynamic_prompt.format(input=prompt, context=context)

def main():
    pass

if __name__ == "__main__":
    main()