import regex as re


def mask_emails(text: str) -> tuple[str, int]:
    email_pattern = re.compile(
        r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})' # Simple email pattern
    )
    masked_text, num_subs = email_pattern.subn('|||EMAIL_ADDRESS|||', text)
    return masked_text, num_subs


def mask_phone_numbers(text: str) -> tuple[str, int]:
    phone_pattern = re.compile(
        r'(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}' # US phone number pattern
    )
    masked_text, num_subs = phone_pattern.subn('|||PHONE_NUMBER|||', text)
    return masked_text, num_subs


def mask_ipv4_addresses(text: str) -> tuple[str, int]:
    ipv4_pattern = re.compile(
        r'\b((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b'
    )
    masked_text, num_subs = ipv4_pattern.subn('|||IP_ADDRESS|||', text)
    return masked_text, num_subs