make:
    echo "Welcome to Project 'memoized_koinapy_wrapper'"

upload_test_pypi:
    twine check dist/*
    python -m pip install --upgrade twine
    twine upload --repository testpypi dist/*

upload_pypi:
    twine check dist/*
    python -m pip install --upgrade twine
    twine upload dist/* 

ve_memoized_koinapy_wrapper:
    python3 -m venv ve_memoized_koinapy_wrapper
