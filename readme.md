# AdFraud

## Description
A script to automate research on "ad fraud in mobile marketing" using the Google Search API and structure the findings with a language model (LLM). It gathers and organizes information into a hierarchical YAML format, covering key subtopics. The project includes a flow diagram to outline the automation process.

## Key Setup
Add your API key's for Google and Search Engine in the `.key` file
```
GOOGLE_API_KEY = <INSERT KEY>
GOOGLE_SEARCH_ID = <INSERT KEY>
```
## Repo Clone
```
git clone https://github.com/ujitkumar1/AdFraud.git
```
```
cd AdFraud
```
## Installation

To install the required packages and libraries, run the following command in your terminal:

```
pip install -r requirements.txt
```

This command will install all the necessary dependencies listed in the requirements.txt file, allowing you to run the
project without any issues.

### Usage:

1. Start the application by running the following command in the project directory:

```
python app.py
```
or 

```
python3 app.py
```
### Workflow:
1. **Google Scrapping**
![google_flow.png](flow_diagram%2Fgoogle_flow.png)
2. **LLM Processing**
![llm_flow.png](flow_diagram%2Fllm_flow.png)

### Contact:

**Name** : Ujit Kumar

**Email** : ujitkumar1@gmail.com

