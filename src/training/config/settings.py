from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    log_path: str

    sentiment_dataset_path: str
    sentiment_model_path: str

    sentiment_target_col: str

    test_size: float
    random_state: int

    hyper_params_yaml_path: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"