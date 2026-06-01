try:
    from myflow.external._scvi import CFJaxSCVI
except ImportError as e:
    raise ImportError(
        "myflow.external requires more dependencies. Please install via pip install 'myflow[external]'"
    ) from e
