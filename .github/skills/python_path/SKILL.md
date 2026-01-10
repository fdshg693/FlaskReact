---
name: python_path
description: how to use global instance **PROJECTPATHS** in this project
---

# Skill Instructions

you should know that this project has a global instance named `PROJECTPATHS` that helps manage file paths consistently across the codebase.
You could always import it from `src/config/paths.py` like this:

```python
# no need to initialize, importing triggers instance creation
from config import PROJECTPATHS

your_path = PROJECTPATHS.some_path_attribute
``` 

you could find all the path attributes in `src/config/paths.txt` file.
If `src/config/paths.txt` file seems outdated, you can run `uv run tests/config/test_paths.py` to regenerate it.