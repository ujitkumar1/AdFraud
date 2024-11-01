from src.google_search.main import run_google_search
from src.llm_process.main import run_llm_process


def run_process():
    run_google_search()
    run_llm_process()


if __name__ == '__main__':
    run_process()
