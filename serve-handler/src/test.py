import os
from handler import GPT2Handler


class MockConfig:

    manifest = {
        "model": {
            "serializedFile": "../latest.pth",
            "modelFile": "model.py",
        }
    }
    system_properties = {
        "model_dir": "/home/agent/dev/repos/souls-gen/serve-handler/src"
    }


if __name__ == "__main__":
    model = GPT2Handler()
    config = MockConfig()
    model.initialize(config)
    print(model.inference(None))
