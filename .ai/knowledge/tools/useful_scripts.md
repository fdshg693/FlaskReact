# Scripts available for AI

## Place
All scripts are located in the `.ai/scripts` directory

## List of Useful Scripts
- `.ai/scripts/choice_selector.py`: 
    - A script to prompt the user to select one option from a list of choices.
    - Useful when human evaluation is needed(e.g., judging implementaion plan options).
- `.ai/scripts/http_request.py`: 
    - A script to make HTTP requests and return the response.This is useful for testing APIs.
    - Especially recommended for windows environments where curl is unstable.
- `.ai/scripts/procman.py`: 
    - A script to manage background processes.
    - Can start, stop, and check the status of processes.
    - Always recomended when double or more processes are needed.
        - frontend/backend servers starting. 
        - server starting, and API testing.