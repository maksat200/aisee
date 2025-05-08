import bcrypt
from sqlalchemy import Boolean, Column, String, Uuid, JSON

# from sqlalchemy.orm import relationship

from src.core.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Uuid, primary_key=True, unique=True)

    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)

    email = Column(String)
    password = Column(String)

    subscription_type = Column(String, nullable=True, default="None")

    is_paid = Column(Boolean, default=False)
    is_superuser = Column(Boolean, default=False)

    other_data = Column(JSON, nullable=True)

    def set_password(self, password: str):

        self.password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def check_password(self, password: str) -> bool:
        return (
            bcrypt.checkpw(password.encode(), self.password.encode())
            if self.password
            else False
        )

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, is_paid={self.is_paid}, is_superuser={self.is_superuser})>"
