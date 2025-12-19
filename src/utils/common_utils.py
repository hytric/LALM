import os
import sys
import importlib
import yaml

def load_config(config_path):
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_obj_from_str(string, reload=False):
    """
    Get object from string path (e.g., "src.LALM.LALMModel")
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = __import__(module, fromlist=[cls])
        importlib.reload(module_imp)
    return getattr(__import__(module, fromlist=[cls]), cls)

def instantiate_from_config(config):
    """
    Instantiate class from config dictionary.
    Supports nested configs by recursively processing params.
    """
    # Config 복사 (원본 보존)
    config = config.copy()
    
    # 클래스 이름 추출 (target 또는 class_name)
    class_name = config.pop("target", None) or config.pop("class_name", None)
    if class_name is None:
        raise ValueError("Config must contain 'target' or 'class_name' key")
    
    # 전체 경로로 클래스 가져오기
    cls = get_obj_from_str(class_name)
    
    # params가 있으면 params를 사용, 없으면 config의 나머지를 사용
    if "params" in config:
        params = config["params"].copy()
    else:
        params = config.copy()
    
    # params 내부의 nested config를 재귀적으로 처리
    for key, value in params.items():
        if isinstance(value, dict) and ("target" in value or "class_name" in value):
            params[key] = instantiate_from_config(value)
    
    # 클래스 인스턴스 생성
    return cls(**params)