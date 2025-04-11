import pydantic 
from pydantic import BaseModel, ConfigDict
from typing import Optional



class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    sub: str = None