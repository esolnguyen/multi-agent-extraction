from decimal import Decimal
from pydantic import BaseModel


class UserInput(BaseModel):
    page_numbers: list[int] | None = None
    instructions: str | None = None


class LLMConfigOption(BaseModel):
    max_output_tokens: int | None = None
    thinking_budget: int | None = None
    temperature: Decimal | None = None
    top_k: int | None = None
    top_p: Decimal | None = None
    seed: int | None = None


class FileInfo(BaseModel):
    original_name: str
    name: str
    extension: str
    mime: str
    no_pages: int
    size_bytes: int
    width: int | None = None
    height: int | None = None


class UserInfo(BaseModel):
    id: str
    email: str
    first_name: str | None = None
    last_name: str | None = None
    status: str


class OrganisationInfo(BaseModel):
    id: str
    name: str
    country: str | None = None
    city: str | None = None
    status: str
    type: str
