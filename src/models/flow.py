from typing import Any
from pydantic import BaseModel
from .types import LLMConfigOption


class OCRFlowStepConfig(BaseModel):
    provider: str
    model_name: str
    ocr_options: dict[str, Any] | None = None


class AgentExecution(BaseModel):
    domain_key: str
    user_prompt: str
    response_schema: dict[str, Any]
    system_prompt: str | None = None
    model_name: str | None = None
    config_options: LLMConfigOption | None = None


class LLMFlowStepConfig(BaseModel):
    provider: str
    system_prompt: str
    user_prompt: str
    model_name: str
    config_options: LLMConfigOption | None = None
    response_schema: dict[str, Any]
    executions: list[AgentExecution] | None = None


class FlowStep(BaseModel):
    step_order: int
    step_type: str
    llm_step_config: LLMFlowStepConfig | None = None
    ocr_step_config: OCRFlowStepConfig | None = None


class FlowConfiguration(BaseModel):
    id: str = ""
    doctype_id: str
    flow_base_id: str
    flow_type: str
    version: int
    steps: list[FlowStep]
    is_active: bool = True
