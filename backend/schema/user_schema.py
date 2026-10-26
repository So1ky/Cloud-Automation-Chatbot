from datetime import datetime
from pydantic import BaseModel, field_validator, EmailStr

class UserCreate(BaseModel):
    email: EmailStr
    password1: str
    password2: str

    @field_validator('email', 'password1', 'password2')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v

    @field_validator('password2')
    def passwords_match(cls, v, info):
        if 'password1' in info.data and v != info.data['password1']:
            raise ValueError('비밀번호가 일치하지 않습니다')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

    @field_validator('email', 'password')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('빈 값은 허용되지 않습니다.')
        return v

class UserResponse(BaseModel):
    id: int
    email: str
    created_at: datetime

    class Config:
        from_attributes = True
