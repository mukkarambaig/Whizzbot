from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

examples = [{
    "input":
    "What are the guidelines for overtime pay?",
    "context":
    "The document states that employees are eligible for overtime pay at a rate of 1.5 times their regular hourly rate for hours worked beyond 40 in a week.",
    "output": "Employees receive overtime pay at 1.5 times the regular rate after 40 hours of work in a week."
}, {
    "input":
    "How does the company support employee professional development?",
    "context":
    "The company's document indicates it offers a $1000 annual allowance for professional development courses for each employee.",
    "output":
    '''
        * The company provides a $1000 annual allowance for professional development.
        * This allowance can be used for courses to enhance professional skills.'''
}, {
    "input":
    "Can employees work remotely from another country?",
    "context":
    "The document does not provide information regarding the company's policy on remote work from another country.",
    "output": "I apologize, but the provided document does not contain information on remote work from another country. Please provide a relevant document or context."
}, {
    "input":
    "What is the company's policy on sick leave?",
    "context":
    "According to the document, the company offers 10 days of paid sick leave annually to its employees.",
    "output":
    '''
        * Employees are entitled to 10 days of paid sick leave annually.
        * This benefit is designed to support employees during health-related absences.'''
}, {
    "input":
    "Describe the company's employee wellness program.",
    "context":
    "The document details a wellness program that includes free gym memberships, monthly wellness workshops, and annual health screenings.",
    "output":
    '''
        * The company offers a comprehensive wellness program.
        * It includes free gym memberships, monthly wellness workshops, and annual health screenings.
        * These initiatives aim to improve overall employee health and well-being.'''
}, {
    "input":
    "What is the company's policy on annual leave?",
    "context":
    "The company policy, as stated in the document, grants employees 20 days of paid annual leave, which they can accrue over the year and must use within the calendar year or forfeit without compensation.",
    "output":
    '''
        * The company provides 20 days of paid annual leave.
        * Leave can be accrued throughout the year.
        * Accrued leave must be used within the same calendar year.
        * Any unused leave is forfeited at the end of the year.
        * There is no compensation for unused leave.
        * Employees are encouraged to plan and utilize their leave.
        * The policy ensures employees have time for rest and recuperation.
        * It's important to schedule leave to avoid losing it.
        * The policy aims to balance work and personal time.
        * Employees should monitor their leave balance to maximize benefits.'''
}]


def prompt_generator(
    conv: str,
    prompt: str,
    context: str,
) -> str:
    prompt_template = PromptTemplate(
        input_variables=["input", "context", "output"],
        template="input: {input}\n\ncontext: {context}\n\noutput: {output}")

    example_selector = LengthBasedExampleSelector(
        examples=examples, example_prompt=prompt_template, max_length=2000)

    dynamic_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=prompt_template,
        prefix=
        """You are an advanced AI assistant designed to function as an HR manager, with complete knowledge of a specific provided document. Your task is to provide clear and understandable answers to queries based on this document's content. Adhere to these guidelines when responding:

         - Rely STRICTLY solely on the information from the provided document to answer questions. If the document does not contain the necessary information, apologize and clarify that the relevant context is missing.
         - Keep your answers concise, using no more than TEN sentence!
         - Format your answers in BULLET POINTS to ensure they are easy to read and understand, even for non-English speakers.
         
         Use the following examples to guide your responses:
         """,
        suffix="conversation history: \n{conversation}\n\nuser wants know about: {input}\n\ncontext: {context}\n\noutput:",
        input_variables=["conversation" ,"input", "context"],
    )

    return dynamic_prompt.format(conversation=conv, input=prompt, context=context)

# NOTE: The following code is only for LLMChain initialization
def prompt_template_generator() -> str:
    prompt_template = PromptTemplate(
        input_variables=["input", "context", "output"],
        template="input: {input}\n\ncontext: {context}\n\noutput: {output}"
    )

    example_selector = LengthBasedExampleSelector(
        examples=examples, example_prompt=prompt_template, max_length=2000
    )

    dynamic_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=prompt_template,
        prefix=
        """You are an advanced AI assistant designed to function as an HR manager, with complete knowledge of a specific provided document. Your task is to provide clear and understandable answers to queries based on this document's content. Adhere to these guidelines when responding:

        - Rely STRICTLY solely on the information from the provided document to answer questions. If the document does not contain the necessary information, apologize and clarify that the relevant context is missing.
        - Keep your answers concise, using no more than TEN sentences!
        - Format your answers in BULLET POINTS to ensure they are easy to read and understand, even for non-English speakers.
        
        Use the following examples to guide your responses:
        """,
        suffix="input: {input}\n\ncontext: {context}\n\noutput:",
        input_variables=["input", "context"],
    )

    return dynamic_prompt



def main():
    from icecream import ic
    prompt = "What are the guidelines for overtime pay?"
    context = "The document states that employees are eligible for overtime pay at a rate of 1.5 times their regular hourly rate for hours worked beyond 40 in a week."
    conv = "How may I assist you today?"
    ic(prompt_generator(conv, prompt, context))


if __name__ == "__main__":
    main()
