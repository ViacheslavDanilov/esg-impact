from pydantic import BaseModel


class TrainingSettings(BaseModel):
    """Configuration settings for model training."""

    # TODO: add more parameters
    db_host: str = "test"
    db_port: int = 123
    db_name: str = "test"
    db_user: str = "test"


class Settings(BaseModel):
    """Main configuration class."""

    train: TrainingSettings = TrainingSettings()


cfg = Settings()
