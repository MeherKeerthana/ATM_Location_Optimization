import importlib.util

spec = importlib.util.spec_from_file_location("verify_1", "verify (1).py")
verify_1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_1)

if __name__ == "__main__":
    verify_1.verify_environment()
    verify_1.test_pipeline()
