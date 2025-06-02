from app import FinancialLLM
from web_scrape import Scraper


def main():
    scraper = Scraper()
    scraper.scrape_and_save()

    finance_llm = FinancialLLM()
    finance_llm.get_tools(db_collection="scraped_news", data_folder="data/scraped_news")
    finance_llm.launch_ui()



if __name__ == "__main__":
    main()
