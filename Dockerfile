    # Use an official Python runtime as a parent image
    FROM python:3.11-bookworm

    RUN useradd -m streamlit

    # Set the working directory in the container to /app
    WORKDIR /app
    ADD requirements.streamlit.txt /app
    # Install any needed packages specified in requirements.txt
    RUN pip install --upgrade pip
    RUN pip install --no-cache-dir -r requirements.streamlit.txt
    RUN ulimit -n 10240

    # Add the current directory contents into the container at /app
    ADD .streamlit/ /app/.streamlit
    ADD *.py /app
    ADD ai/*.py /app/ai/
    ADD ui/*.py /app/ui/

    RUN chown -R streamlit:streamlit /app


    USER streamlit

    EXPOSE 8501

    # Run app.py when the container launches
    CMD ["streamlit", "run", "--ui.hideTopBar=True", "--global.developmentMode=False", "main.py"]